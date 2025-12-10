"""
Improved RoBERTa + CRF Sequence Model with Advanced Training Techniques

Improvements:
1. Class-weighted loss for minority emotions
2. Dropout regularization
3. Learning rate scheduling with warmup
4. Gradient clipping
5. Early stopping on validation macro F1
6. Data augmentation via oversampling minority classes
7. Larger model capacity
8. Better transition regularization
"""

import argparse
import json
import math
import pathlib
from collections import Counter, defaultdict
from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from sklearn.metrics import classification_report, f1_score, accuracy_score
from transformers import AutoModel, AutoTokenizer, get_linear_schedule_with_warmup
from torchcrf import CRF


PROJECT_ROOT = pathlib.Path(__file__).resolve().parents[1]
PROC = PROJECT_ROOT / "data_proc"
MODELS_DIR = PROJECT_ROOT / "models"
MODELS_DIR.mkdir(exist_ok=True)


EMOTION_LABELS = [
    "anger",
    "disgust",
    "fear",
    "happiness",
    "no_emotion",
    "sadness",
    "surprise",
]
LABEL2ID = {lbl: i for i, lbl in enumerate(EMOTION_LABELS)}
ID2LABEL = {i: lbl for lbl, i in LABEL2ID.items()}


@dataclass
class ImprovedSeqConfig:
    model_name: str = "roberta-base"
    max_utt_length: int = 128  # Increased from 64
    batch_size: int = 8  # Increased for GPU
    num_epochs: int = 10  # Increased from 3
    learning_rate: float = 3e-5  # Slightly higher
    weight_decay: float = 0.01
    warmup_ratio: float = 0.1
    transition_reg_lambda: float = 0.05  # Reduced for more flexibility
    dropout: float = 0.3  # Added dropout
    gradient_clip: float = 1.0
    patience: int = 3  # Early stopping patience
    device: str = "cpu"  # Force CPU due to MPS memory constraints


class DialogDataset(Dataset):
    def __init__(self, dialogues: List[List[str]], labels: List[List[int]]):
        self.dialogues = dialogues
        self.labels = labels

    def __len__(self):
        return len(self.dialogues)

    def __getitem__(self, idx):
        return self.dialogues[idx], self.labels[idx]


def load_dailydialog_with_augmentation():
    """Load DailyDialog and oversample minority emotion dialogues."""
    dd_df = pd.read_csv(PROC / "dailydialog_utterances.csv")
    
    dialogues = []
    labels = []
    
    for dialog_id, group in dd_df.groupby("dialog_id"):
        group = group.sort_values("turn_id")
        dialog_texts = group["utterance"].tolist()
        dialog_labels = group["emotion_id"].tolist()
        
        dialogues.append(dialog_texts)
        labels.append(dialog_labels)
    
    # Split into train/val/test
    n = len(dialogues)
    train_size = int(0.8 * n)
    val_size = int(0.1 * n)
    
    train_dialogues = dialogues[:train_size]
    train_labels = labels[:train_size]
    
    val_dialogues = dialogues[train_size:train_size + val_size]
    val_labels = labels[train_size:train_size + val_size]
    
    test_dialogues = dialogues[train_size + val_size:]
    test_labels = labels[train_size + val_size:]
    
    # Oversample minority emotions in training
    train_dialogues, train_labels = oversample_minority_emotions(
        train_dialogues, train_labels
    )
    
    return (train_dialogues, train_labels), (val_dialogues, val_labels), (test_dialogues, test_labels)


def oversample_minority_emotions(dialogues, labels, target_ratio=0.3):
    """Oversample dialogues containing minority emotions."""
    minority_emotions = {0, 1, 2, 5}  # anger, disgust, fear, sadness
    
    minority_dialogues = []
    minority_labels = []
    
    for dialog, labs in zip(dialogues, labels):
        if any(l in minority_emotions for l in labs):
            minority_dialogues.append(dialog)
            minority_labels.append(labs)
    
    # Calculate how many times to repeat minority dialogues
    n_majority = len(dialogues) - len(minority_dialogues)
    n_minority = len(minority_dialogues)
    
    if n_minority > 0:
        repeat_times = max(1, int((n_majority * target_ratio) / n_minority) - 1)
        
        augmented_dialogues = dialogues + minority_dialogues * repeat_times
        augmented_labels = labels + minority_labels * repeat_times
        
        return augmented_dialogues, augmented_labels
    
    return dialogues, labels


def compute_class_weights(dialog_labels: List[List[int]], num_labels: int) -> torch.Tensor:
    """Compute class weights for imbalanced emotions."""
    counts = Counter()
    for labels in dialog_labels:
        counts.update(labels)
    
    total = sum(counts.values())
    weights = torch.ones(num_labels)
    
    for i in range(num_labels):
        if counts[i] > 0:
            weights[i] = total / (num_labels * counts[i])
    
    # Normalize weights
    weights = weights / weights.sum() * num_labels
    
    return weights


def compute_transition_prior(dialog_labels: List[List[int]], num_labels: int) -> torch.Tensor:
    """Compute empirical transition probabilities."""
    trans_counts = torch.zeros((num_labels, num_labels))
    
    for labels in dialog_labels:
        for i in range(len(labels) - 1):
            trans_counts[labels[i], labels[i + 1]] += 1
    
    # Add smoothing
    trans_counts += 0.1
    
    # Normalize
    row_sums = trans_counts.sum(dim=1, keepdim=True)
    trans_probs = trans_counts / row_sums
    
    return trans_probs


def collate_dialog_batch(batch, tokenizer, max_length, device):
    """Collate dialogues into a batch."""
    all_utterances = []
    all_labels = []
    dialog_lens = []
    
    for dialog, labels in batch:
        all_utterances.extend(dialog)
        all_labels.append(torch.tensor(labels, dtype=torch.long))
        dialog_lens.append(len(dialog))
    
    # Tokenize all utterances
    enc = tokenizer(
        all_utterances,
        padding=True,
        truncation=True,
        max_length=max_length,
        return_tensors="pt",
    )
    enc = {k: v.to(device) for k, v in enc.items()}
    
    # Build label tensor and mask
    batch_size = len(batch)
    max_dialog_len = max(dialog_lens)
    label_tensor = torch.full((batch_size, max_dialog_len), -100, dtype=torch.long, device=device)
    hidden_mask = torch.zeros((batch_size, max_dialog_len), dtype=torch.bool, device=device)
    
    utt_indices = []
    idx = 0
    for b, (labels, L) in enumerate(zip(all_labels, dialog_lens)):
        label_tensor[b, :L] = labels.to(device)
        hidden_mask[b, :L] = True
        utt_indices.extend([(b, t) for t in range(L)])
        idx += L
    
    enc["utt_indices"] = utt_indices
    
    return enc, label_tensor, hidden_mask


class ImprovedRobertaCRFModel(nn.Module):
    def __init__(
        self,
        model_name: str,
        num_labels: int,
        transition_prior: torch.Tensor,
        class_weights: torch.Tensor,
        reg_lambda: float = 0.05,
        dropout: float = 0.3,
    ):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(model_name)
        hidden_size = self.encoder.config.hidden_size
        
        # Add dropout for regularization
        self.dropout = nn.Dropout(dropout)
        
        # Larger classifier with intermediate layer
        self.classifier = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size // 2, num_labels),
        )
        
        self.crf = CRF(num_tags=num_labels, batch_first=True)
        self.reg_lambda = reg_lambda
        
        self.register_buffer("transition_prior", transition_prior)
        self.register_buffer("class_weights", class_weights)

    def forward(self, encodings, label_tensor=None, dialog_mask=None):
        # Extract helper data
        utt_indices = encodings["utt_indices"]
        enc_inputs = {k: v for k, v in encodings.items() if k != "utt_indices"}
        
        # Encode utterances
        outputs = self.encoder(**enc_inputs)
        cls_hidden = outputs.last_hidden_state[:, 0, :]
        
        # Apply dropout
        cls_hidden = self.dropout(cls_hidden)
        
        # Map to dialog structure
        batch_size = dialog_mask.size(0)
        max_dialog_len = dialog_mask.size(1)
        H = cls_hidden.size(-1)
        dialog_repr = cls_hidden.new_zeros((batch_size, max_dialog_len, H))
        
        for i, (b, t) in enumerate(utt_indices):
            dialog_repr[b, t] = cls_hidden[i]
        
        # Get emissions
        emissions = self.classifier(dialog_repr)
        
        if label_tensor is not None:
            # Training mode
            mask = (label_tensor != -100) & dialog_mask
            
            # Sanitize labels for CRF
            tags = label_tensor.clone()
            tags[~mask] = 0
            
            # CRF loss
            nll = -self.crf(emissions, tags, mask=mask, reduction="mean")
            
            # Transition regularization
            trans_reg = self.reg_lambda * torch.mean(
                (self.crf.transitions - self.transition_prior) ** 2
            )
            
            # Class-weighted emission loss (auxiliary)
            flat_emissions = emissions[mask]
            flat_labels = tags[mask]
            ce_loss = F.cross_entropy(
                flat_emissions,
                flat_labels,
                weight=self.class_weights,
                reduction="mean",
            )
            
            # Combined loss
            total_loss = nll + trans_reg + 0.1 * ce_loss
            
            return total_loss
        else:
            # Inference mode
            best_paths = self.crf.decode(emissions, mask=dialog_mask)
            return best_paths


def train_improved_model(cfg: ImprovedSeqConfig):
    """Train the improved sequence model."""
    print("=" * 70)
    print("Training Improved RoBERTa + CRF Sequence Model")
    print("=" * 70)
    
    # Load data
    print("\nLoading and augmenting data...")
    (train_dialogues, train_labels), (val_dialogues, val_labels), (test_dialogues, test_labels) = (
        load_dailydialog_with_augmentation()
    )
    
    print(f"Train dialogues: {len(train_dialogues)}")
    print(f"Val dialogues: {len(val_dialogues)}")
    print(f"Test dialogues: {len(test_dialogues)}")
    
    # Compute class weights and transition prior
    print("\nComputing class weights and transition prior...")
    class_weights = compute_class_weights(train_labels, len(EMOTION_LABELS))
    transition_prior = compute_transition_prior(train_labels, len(EMOTION_LABELS))
    
    print(f"Class weights: {class_weights}")
    
    # Create datasets
    train_dataset = DialogDataset(train_dialogues, train_labels)
    val_dataset = DialogDataset(val_dialogues, val_labels)
    test_dataset = DialogDataset(test_dialogues, test_labels)
    
    # Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(cfg.model_name)
    
    # Dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=cfg.batch_size,
        shuffle=True,
        collate_fn=lambda b: collate_dialog_batch(b, tokenizer, cfg.max_utt_length, cfg.device),
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=cfg.batch_size,
        shuffle=False,
        collate_fn=lambda b: collate_dialog_batch(b, tokenizer, cfg.max_utt_length, cfg.device),
    )
    
    # Model
    print(f"\nInitializing model on {cfg.device}...")
    model = ImprovedRobertaCRFModel(
        model_name=cfg.model_name,
        num_labels=len(EMOTION_LABELS),
        transition_prior=transition_prior.to(cfg.device),
        class_weights=class_weights.to(cfg.device),
        reg_lambda=cfg.transition_reg_lambda,
        dropout=cfg.dropout,
    ).to(cfg.device)
    
    # Optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=cfg.learning_rate,
        weight_decay=cfg.weight_decay,
    )
    
    # Learning rate scheduler
    total_steps = len(train_loader) * cfg.num_epochs
    warmup_steps = int(cfg.warmup_ratio * total_steps)
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_steps,
    )
    
    # Training loop with early stopping
    best_val_f1 = 0.0
    patience_counter = 0
    
    print(f"\nStarting training for {cfg.num_epochs} epochs...")
    print(f"Warmup steps: {warmup_steps}, Total steps: {total_steps}")
    print("=" * 70)
    
    for epoch in range(cfg.num_epochs):
        # Training
        model.train()
        train_loss = 0.0
        
        for batch_idx, (enc, label_tensor, dialog_mask) in enumerate(train_loader):
            optimizer.zero_grad()
            
            loss = model(enc, label_tensor=label_tensor, dialog_mask=dialog_mask)
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.gradient_clip)
            
            optimizer.step()
            scheduler.step()
            
            train_loss += loss.item()
            
            if (batch_idx + 1) % 100 == 0:
                print(f"Epoch {epoch + 1}/{cfg.num_epochs} - Batch {batch_idx + 1}/{len(train_loader)} - Loss: {loss.item():.4f}")
        
        avg_train_loss = train_loss / len(train_loader)
        
        # Validation
        model.eval()
        val_preds = []
        val_true = []
        
        with torch.no_grad():
            for enc, label_tensor, dialog_mask in val_loader:
                predictions = model(enc, dialog_mask=dialog_mask)
                
                # Flatten predictions and labels
                for b, pred_seq in enumerate(predictions):
                    true_seq = label_tensor[b].cpu().tolist()
                    true_seq = [l for l in true_seq if l != -100]
                    
                    val_preds.extend(pred_seq[:len(true_seq)])
                    val_true.extend(true_seq)
        
        val_f1 = f1_score(val_true, val_preds, average="macro")
        val_acc = accuracy_score(val_true, val_preds)
        
        print(f"\nEpoch {epoch + 1}/{cfg.num_epochs}")
        print(f"  Train Loss: {avg_train_loss:.4f}")
        print(f"  Val Accuracy: {val_acc:.4f}")
        print(f"  Val Macro F1: {val_f1:.4f}")
        
        # Early stopping
        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            patience_counter = 0
            
            # Save best model
            model_path = MODELS_DIR / "roberta_crf_improved_best.pt"
            torch.save(model.state_dict(), model_path)
            print(f"  ✓ New best model saved (F1: {val_f1:.4f})")
        else:
            patience_counter += 1
            print(f"  No improvement (patience: {patience_counter}/{cfg.patience})")
            
            if patience_counter >= cfg.patience:
                print(f"\nEarly stopping triggered after {epoch + 1} epochs")
                break
        
        print("-" * 70)
    
    # Load best model for testing
    print("\nLoading best model for testing...")
    model.load_state_dict(torch.load(MODELS_DIR / "roberta_crf_improved_best.pt"))
    model.eval()
    
    # Test evaluation
    test_loader = DataLoader(
        test_dataset,
        batch_size=cfg.batch_size,
        shuffle=False,
        collate_fn=lambda b: collate_dialog_batch(b, tokenizer, cfg.max_utt_length, cfg.device),
    )
    
    test_preds = []
    test_true = []
    
    with torch.no_grad():
        for enc, label_tensor, dialog_mask in test_loader:
            predictions = model(enc, dialog_mask=dialog_mask)
            
            for b, pred_seq in enumerate(predictions):
                true_seq = label_tensor[b].cpu().tolist()
                true_seq = [l for l in true_seq if l != -100]
                
                test_preds.extend(pred_seq[:len(true_seq)])
                test_true.extend(true_seq)
    
    # Generate report
    report = classification_report(
        test_true,
        test_preds,
        target_names=EMOTION_LABELS,
        digits=4,
    )
    
    acc = accuracy_score(test_true, test_preds)
    macro_f1 = f1_score(test_true, test_preds, average="macro")
    
    print("\n" + "=" * 70)
    print("FINAL TEST RESULTS")
    print("=" * 70)
    print(report)
    print(f"\nAccuracy: {acc:.4f}")
    print(f"Macro F1: {macro_f1:.4f}")
    
    # Save report
    report_path = MODELS_DIR / "roberta_crf_improved_evaluation_report.txt"
    with open(report_path, "w") as f:
        f.write("Improved RoBERTa + CRF Sequence Model\n")
        f.write("=" * 72 + "\n\n")
        f.write("Improvements:\n")
        f.write("- Class-weighted loss for minority emotions\n")
        f.write("- Dropout regularization (0.3)\n")
        f.write("- Learning rate scheduling with warmup\n")
        f.write("- Gradient clipping\n")
        f.write("- Early stopping on validation macro F1\n")
        f.write("- Oversampling minority emotion dialogues\n")
        f.write("- Larger classifier with intermediate layer\n")
        f.write("- Auxiliary cross-entropy loss\n\n")
        f.write("=" * 72 + "\n\n")
        f.write(report)
        f.write(f"\n\nAccuracy: {acc:.4f}\n")
        f.write(f"Macro F1: {macro_f1:.4f}\n")
    
    print(f"\n✓ Evaluation report saved to: {report_path}")
    print(f"✓ Best model saved to: {MODELS_DIR / 'roberta_crf_improved_best.pt'}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-name", type=str, default="roberta-base")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--lr", type=float, default=3e-5)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--max-utt-length", type=int, default=128)
    parser.add_argument("--dropout", type=float, default=0.3)
    parser.add_argument("--transition-reg-lambda", type=float, default=0.05)
    
    args = parser.parse_args()
    
    cfg = ImprovedSeqConfig(
        model_name=args.model_name,
        num_epochs=args.epochs,
        learning_rate=args.lr,
        batch_size=args.batch_size,
        max_utt_length=args.max_utt_length,
        dropout=args.dropout,
        transition_reg_lambda=args.transition_reg_lambda,
    )
    
    train_improved_model(cfg)


if __name__ == "__main__":
    main()
