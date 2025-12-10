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
from torch.utils.data import DataLoader, Dataset
from sklearn.metrics import classification_report, f1_score, accuracy_score
from sklearn.model_selection import train_test_split
from transformers import AutoModel, AutoTokenizer
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
class SeqConfig:
    model_name: str = "roberta-base"
    max_utt_length: int = 64
    batch_size: int = 4
    num_epochs: int = 3
    learning_rate: float = 2e-5
    transition_reg_lambda: float = 0.1
    device: str = "cuda" if torch.cuda.is_available() else "cpu"


class DialogDataset(Dataset):
    """Dialogue-level dataset: each item is (list of utterances, list of label_ids)."""

    def __init__(self, dialogs: List[List[str]], labels: List[List[int]]):
        self.dialogs = dialogs
        self.labels = labels

    def __len__(self) -> int:
        return len(self.dialogs)

    def __getitem__(self, idx: int):
        return {
            "utterances": self.dialogs[idx],
            "labels": torch.tensor(self.labels[idx], dtype=torch.long),
        }


def build_dialog_sequences(df: pd.DataFrame) -> Tuple[List[List[str]], List[List[int]]]:
    dialogs: Dict[int, List[Tuple[int, str, int]]] = defaultdict(list)
    for _, row in df.iterrows():
        did = int(row["dialog_id"])
        tid = int(row["turn_id"])
        utt = str(row["utterance"])
        lbl = LABEL2ID[str(row["emotion"])]
        dialogs[did].append((tid, utt, lbl))

    dialog_texts: List[List[str]] = []
    dialog_labels: List[List[int]] = []
    for did, turns in dialogs.items():
        turns_sorted = sorted(turns, key=lambda x: x[0])
        utts = [t[1] for t in turns_sorted]
        lbs = [t[2] for t in turns_sorted]
        dialog_texts.append(utts)
        dialog_labels.append(lbs)
    return dialog_texts, dialog_labels


def collate_batch(batch, tokenizer, max_utt_length: int):
    # batch: list of {utterances: List[str], labels: tensor[T]}
    all_utterances = []
    dialog_lens = []
    all_labels = []
    for item in batch:
        utts = item["utterances"]
        labels = item["labels"]
        dialog_lens.append(len(utts))
        all_utterances.extend(utts)
        all_labels.append(labels)

    # Tokenize all utterances in flat form
    enc = tokenizer(
        all_utterances,
        padding=True,
        truncation=True,
        max_length=max_utt_length,
        return_tensors="pt",
    )

    # Pack back into dialogues
    max_dialog_len = max(dialog_lens)
    batch_size = len(batch)
    hidden_mask = torch.zeros(batch_size, max_dialog_len, dtype=torch.bool)
    label_tensor = torch.full((batch_size, max_dialog_len), fill_value=-100, dtype=torch.long)

    # we will later reshape token encodings to (num_utts, seq_len,...)
    # track index
    idx = 0
    utt_indices = []  # start,end for each utterance
    for b, (labels, L) in enumerate(zip(all_labels, dialog_lens)):
        label_tensor[b, :L] = labels
        hidden_mask[b, :L] = True
        utt_indices.extend([(b, t) for t in range(L)])
        idx += L

    return enc, label_tensor, hidden_mask, utt_indices


class RobertaCRFSequenceModel(nn.Module):
    def __init__(self, model_name: str, num_labels: int, transition_prior: torch.Tensor, reg_lambda: float = 0.1):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(model_name)
        hidden_size = self.encoder.config.hidden_size
        self.classifier = nn.Linear(hidden_size, num_labels)
        self.crf = CRF(num_tags=num_labels, batch_first=True)
        self.reg_lambda = reg_lambda
        # transition_prior: [num_labels, num_labels]
        self.register_buffer("transition_prior", transition_prior)

    def forward(self, encodings, label_tensor=None, dialog_mask=None):
        # encodings: output of tokenizer (flattened utterances) plus helper key 'utt_indices'.
        # Remove helper key before passing kwargs to the encoder.
        utt_indices: List[Tuple[int, int]] = encodings["utt_indices"]
        enc_inputs = {k: v for k, v in encodings.items() if k != "utt_indices"}

        outputs = self.encoder(**enc_inputs)
        # use [CLS] token representation
        cls_hidden = outputs.last_hidden_state[:, 0, :]  # (N_utts, H)
        # We need to map these back to (B, T, H) using dialog_mask
        # enc_inputs["input_ids"].shape[0] == total_utterances in batch
        batch_size = dialog_mask.size(0)
        max_dialog_len = dialog_mask.size(1)
        H = cls_hidden.size(-1)
        dialog_repr = cls_hidden.new_zeros((batch_size, max_dialog_len, H))
        for i, (b, t) in enumerate(utt_indices):
            dialog_repr[b, t] = cls_hidden[i]

        emissions = self.classifier(dialog_repr)  # (B, T, num_labels)

        if label_tensor is not None:
            # mask positions where label != -100
            mask = (label_tensor != -100) & dialog_mask
            # torchcrf expects all tag indices to be valid (0..num_labels-1), even
            # where mask is False. Replace -100 labels with a safe value (0) before
            # calling CRF so indexing stays in range.
            tags = label_tensor.clone()
            tags[~mask] = 0

            # CRF expects mask as bool
            nll = -self.crf(emissions, tags, mask=mask, reduction="mean")
            # Transition regularization
            reg = self.reg_lambda * torch.mean((self.crf.transitions - self.transition_prior) ** 2)
            loss = nll + reg
            return loss
        else:
            best_paths = self.crf.decode(emissions, mask=dialog_mask)
            return best_paths


def compute_transition_prior(train_dialog_labels: List[List[int]], num_labels: int) -> torch.Tensor:
    counts = np.ones((num_labels, num_labels), dtype=np.float32)  # Laplace smoothing
    for seq in train_dialog_labels:
        for i in range(len(seq) - 1):
            a, b = seq[i], seq[i + 1]
            counts[a, b] += 1.0
    probs = counts / counts.sum(axis=1, keepdims=True)
    # Use log-probabilities as a soft prior
    logp = np.log(probs + 1e-8)
    return torch.tensor(logp, dtype=torch.float32)


def train_sequence_model(cfg: SeqConfig):
    print("Loading original DailyDialog utterances with dialogue structure...")
    df = pd.read_csv(PROC / "dailydialog_utterances.csv")
    # Filter to emotions we care about
    df = df[df["emotion"].isin(EMOTION_LABELS)].reset_index(drop=True)

    dialogs, labels = build_dialog_sequences(df)
    dialog_ids = np.arange(len(dialogs))
    train_ids, temp_ids = train_test_split(dialog_ids, test_size=0.2, random_state=42)
    val_ids, test_ids = train_test_split(temp_ids, test_size=0.5, random_state=42)

    def subset(ids):
        return [dialogs[i] for i in ids], [labels[i] for i in ids]

    train_dialogs, train_labels = subset(train_ids)
    val_dialogs, val_labels = subset(val_ids)
    test_dialogs, test_labels = subset(test_ids)

    print(f"Train/Val/Test dialogues: {len(train_dialogs)} / {len(val_dialogs)} / {len(test_dialogs)}")

    tokenizer = AutoTokenizer.from_pretrained(cfg.model_name)

    train_dataset = DialogDataset(train_dialogs, train_labels)
    val_dataset = DialogDataset(val_dialogs, val_labels)
    test_dataset = DialogDataset(test_dialogs, test_labels)

    def collate_fn(batch):
        enc, label_tensor, dialog_mask, utt_indices = collate_batch(
            batch, tokenizer, max_utt_length=cfg.max_utt_length
        )
        enc["utt_indices"] = utt_indices
        return enc, label_tensor, dialog_mask

    train_loader = DataLoader(train_dataset, batch_size=cfg.batch_size, shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=cfg.batch_size, shuffle=False, collate_fn=collate_fn)
    test_loader = DataLoader(test_dataset, batch_size=cfg.batch_size, shuffle=False, collate_fn=collate_fn)

    transition_prior = compute_transition_prior(train_labels, num_labels=len(EMOTION_LABELS))

    model = RobertaCRFSequenceModel(
        model_name=cfg.model_name,
        num_labels=len(EMOTION_LABELS),
        transition_prior=transition_prior,
        reg_lambda=cfg.transition_reg_lambda,
    ).to(cfg.device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.learning_rate)

    for epoch in range(cfg.num_epochs):
        model.train()
        total_loss = 0.0
        for enc, label_tensor, dialog_mask in train_loader:
            enc = {k: v.to(cfg.device) if hasattr(v, "to") else v for k, v in enc.items()}
            label_tensor = label_tensor.to(cfg.device)
            dialog_mask = dialog_mask.to(cfg.device)

            loss = model(enc, label_tensor=label_tensor, dialog_mask=dialog_mask)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
        avg_loss = total_loss / max(1, len(train_loader))
        print(f"Epoch {epoch+1}/{cfg.num_epochs} - Train loss: {avg_loss:.4f}")

    # Evaluation on test set
    model.eval()
    all_true: List[int] = []
    all_pred: List[int] = []
    with torch.no_grad():
        for enc, label_tensor, dialog_mask in test_loader:
            enc = {k: v.to(cfg.device) if hasattr(v, "to") else v for k, v in enc.items()}
            label_tensor = label_tensor.to(cfg.device)
            dialog_mask = dialog_mask.to(cfg.device)

            best_paths = model(enc, label_tensor=None, dialog_mask=dialog_mask)
            # Flatten predictions and labels
            for path, labels_seq, mask_seq in zip(best_paths, label_tensor, dialog_mask):
                for p, y, m in zip(path, labels_seq, mask_seq):
                    if m and y.item() != -100:
                        all_true.append(y.item())
                        all_pred.append(p)

    report = classification_report(
        all_true,
        all_pred,
        target_names=EMOTION_LABELS,
        digits=4,
    )
    acc = accuracy_score(all_true, all_pred)
    macro_f1 = f1_score(all_true, all_pred, average="macro")

    print(report)
    print(f"Accuracy: {acc:.4f}")
    print(f"Macro F1: {macro_f1:.4f}")

    report_path = MODELS_DIR / "roberta_crf_sequence_evaluation_report.txt"
    with open(report_path, "w") as f:
        f.write("Model: RoBERTa + CRF sequence model with transition regularization\n")
        f.write("=" * 72 + "\n\n")
        f.write(report)
        f.write("\n\n")
        f.write(f"Accuracy: {acc:.4f}\n")
        f.write(f"Macro F1: {macro_f1:.4f}\n")

    print(f"Saved sequence model evaluation report to: {report_path}")


def main():
    parser = argparse.ArgumentParser(description="Train RoBERTa+CRF sequence model for emotion classification")
    parser.add_argument("--model-name", type=str, default="roberta-base")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--max-utt-length", type=int, default=64)
    parser.add_argument("--transition-reg-lambda", type=float, default=0.1)
    args = parser.parse_args()

    cfg = SeqConfig(
        model_name=args.model_name,
        num_epochs=args.epochs,
        learning_rate=args.lr,
        batch_size=args.batch_size,
        max_utt_length=args.max_utt_length,
        transition_reg_lambda=args.transition_reg_lambda,
    )

    train_sequence_model(cfg)


if __name__ == "__main__":
    main()
