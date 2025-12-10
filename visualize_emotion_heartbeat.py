"""
Emotion Heartbeat Map Visualization
Visualizes emotion predictions across dialogue utterances as a line plot.
"""

import pathlib
import json
from typing import List, Dict, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
from transformers import AutoTokenizer

# Import the sequence model
import sys
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from src.train_roberta_crf_sequence import (
    RobertaCRFSequenceModel,
    compute_transition_prior,
    EMOTION_LABELS,
    ID2LABEL,
)

PROJECT_ROOT = pathlib.Path(__file__).resolve().parent
PROC = PROJECT_ROOT / "data_proc"
MODELS_DIR = PROJECT_ROOT / "models"


def load_test_dialogues() -> Tuple[List[List[str]], List[List[int]], List[int]]:
    """Load dialogues with utterances and emotion labels from DailyDialog."""
    # Load the original DailyDialog with dialogue structure
    dd_df = pd.read_csv(PROC / "dailydialog_utterances.csv")
    
    # Group by dialog_id
    dialogues = []
    labels = []
    dialog_ids = []
    
    for dialog_id, group in dd_df.groupby("dialog_id"):
        group = group.sort_values("turn_id")
        
        dialog_texts = group["utterance"].tolist()
        dialog_labels = group["emotion_id"].tolist()
        
        dialogues.append(dialog_texts)
        labels.append(dialog_labels)
        dialog_ids.append(dialog_id)
    
    return dialogues, labels, dialog_ids


def predict_dialogue_emotions(
    model: RobertaCRFSequenceModel,
    tokenizer,
    utterances: List[str],
    device: str,
    max_length: int = 64,
) -> List[int]:
    """Predict emotion sequence for a single dialogue."""
    model.eval()
    
    # Tokenize all utterances
    encodings = tokenizer(
        utterances,
        padding=True,
        truncation=True,
        max_length=max_length,
        return_tensors="pt",
    )
    
    # Move to device
    encodings = {k: v.to(device) for k, v in encodings.items()}
    
    # Add batch dimension and utt_indices
    batch_size = 1
    dialog_len = len(utterances)
    dialog_mask = torch.ones((batch_size, dialog_len), dtype=torch.bool, device=device)
    utt_indices = [(0, t) for t in range(dialog_len)]
    encodings["utt_indices"] = utt_indices
    
    with torch.no_grad():
        predictions = model(encodings, dialog_mask=dialog_mask)
    
    # predictions is a list of lists (batch_first=True)
    return predictions[0]


def plot_emotion_heartbeat(
    utterances: List[str],
    true_labels: List[int],
    pred_labels: List[int],
    emotion_names: List[str],
    dialogue_id: str,
    save_path: pathlib.Path,
):
    """
    Create a heartbeat-style line plot showing emotion evolution.
    
    Y-axis: emotion categories (mapped to numeric values)
    X-axis: utterance positions
    """
    # Map emotion IDs to y-axis positions
    emotion_to_y = {i: i for i in range(len(emotion_names))}
    
    # Get y positions for true and predicted
    true_y = [emotion_to_y[label] for label in true_labels]
    pred_y = [emotion_to_y[label] for label in pred_labels]
    
    x = list(range(1, len(utterances) + 1))
    
    # Create figure
    fig, ax = plt.subplots(figsize=(14, 8))
    
    # Plot true emotions (ground truth)
    ax.plot(
        x, true_y, 
        marker='o', 
        markersize=8, 
        linewidth=2.5, 
        color='#2E86AB',
        label='Ground Truth',
        alpha=0.8,
    )
    
    # Plot predicted emotions
    ax.plot(
        x, pred_y, 
        marker='s', 
        markersize=7, 
        linewidth=2, 
        color='#A23B72',
        linestyle='--',
        label='Predicted',
        alpha=0.8,
    )
    
    # Set y-axis labels to emotion names
    ax.set_yticks(range(len(emotion_names)))
    ax.set_yticklabels(emotion_names, fontsize=11)
    
    # Set x-axis
    ax.set_xticks(x)
    ax.set_xticklabels([f"ut{i}" for i in x], fontsize=10)
    ax.set_xlabel("Utterance Position", fontsize=13, fontweight='bold')
    ax.set_ylabel("Emotion", fontsize=13, fontweight='bold')
    
    # Title
    ax.set_title(
        f"Emotion Heartbeat Map - Dialogue {dialogue_id}",
        fontsize=15,
        fontweight='bold',
        pad=20,
    )
    
    # Grid
    ax.grid(True, alpha=0.3, linestyle=':', linewidth=0.8)
    ax.set_axisbelow(True)
    
    # Legend
    ax.legend(loc='upper right', fontsize=11, framealpha=0.9)
    
    # Add utterance text as annotations (optional, for first few)
    max_annotations = min(6, len(utterances))
    for i in range(max_annotations):
        # Truncate long utterances
        text = utterances[i][:40] + "..." if len(utterances[i]) > 40 else utterances[i]
        ax.annotate(
            text,
            xy=(x[i], true_y[i]),
            xytext=(10, 10),
            textcoords='offset points',
            fontsize=8,
            alpha=0.6,
            bbox=dict(boxstyle='round,pad=0.3', facecolor='yellow', alpha=0.3),
        )
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"Saved heartbeat map to: {save_path}")


def main():
    # Configuration
    model_name = "roberta-base"
    device = "cuda" if torch.cuda.is_available() else "cpu"
    max_utt_length = 64
    
    print(f"Using device: {device}")
    
    # Load emotion mapping
    emotion_names = [ID2LABEL[i] for i in sorted(ID2LABEL.keys())]
    num_labels = len(emotion_names)
    
    print(f"Emotion labels: {emotion_names}")
    
    # Load test dialogues
    print("\nLoading dialogues...")
    dialogues, true_labels, dialog_ids = load_test_dialogues()
    print(f"Loaded {len(dialogues)} dialogues")
    
    # Compute transition prior (needed for model)
    # Use a subset of dialogues for transition prior (first 80% as "train")
    print("\nComputing transition prior from dialogue data...")
    train_size = int(0.8 * len(dialogues))
    train_dialog_labels = true_labels[:train_size]
    
    transition_prior = compute_transition_prior(train_dialog_labels, num_labels)
    
    # Load trained model
    model_path = MODELS_DIR / "roberta_crf_sequence_model.pt"
    if not model_path.exists():
        print(f"Error: Model not found at {model_path}")
        print("Please train the sequence model first using train_roberta_crf_sequence.py")
        return
    
    print(f"\nLoading model from {model_path}...")
    model = RobertaCRFSequenceModel(
        model_name=model_name,
        num_labels=num_labels,
        transition_prior=transition_prior,
        reg_lambda=0.1,
    ).to(device)
    
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    # Create output directory for visualizations
    viz_dir = PROJECT_ROOT / "visualizations"
    viz_dir.mkdir(exist_ok=True)
    
    # Select a few interesting dialogues to visualize
    # Criteria: dialogues with diverse emotions and reasonable length
    print("\nSelecting dialogues for visualization...")
    
    selected_indices = []
    for i, (dialog, labels) in enumerate(zip(dialogues, true_labels)):
        # Select dialogues with 4-10 utterances and at least 3 different emotions
        if 4 <= len(dialog) <= 10 and len(set(labels)) >= 3:
            selected_indices.append(i)
            if len(selected_indices) >= 5:  # Visualize 5 dialogues
                break
    
    if not selected_indices:
        # Fallback: just take first 5 dialogues
        selected_indices = list(range(min(5, len(dialogues))))
    
    print(f"Visualizing {len(selected_indices)} dialogues...")
    
    # Generate visualizations
    for idx in selected_indices:
        utterances = dialogues[idx]
        true_labs = true_labels[idx]
        
        print(f"\nDialogue {idx + 1}:")
        print(f"  Utterances: {len(utterances)}")
        print(f"  True emotions: {[emotion_names[l] for l in true_labs]}")
        
        # Predict
        pred_labs = predict_dialogue_emotions(
            model, tokenizer, utterances, device, max_utt_length
        )
        print(f"  Predicted emotions: {[emotion_names[l] for l in pred_labs]}")
        
        # Plot
        actual_dialog_id = dialog_ids[idx]
        save_path = viz_dir / f"emotion_heartbeat_dialogue_{actual_dialog_id}.png"
        plot_emotion_heartbeat(
            utterances=utterances,
            true_labels=true_labs,
            pred_labels=pred_labs,
            emotion_names=emotion_names,
            dialogue_id=str(actual_dialog_id),
            save_path=save_path,
        )
    
    print(f"\n✓ All visualizations saved to: {viz_dir}")
    print("\nYou can now view the emotion heartbeat maps in the visualizations/ folder.")


if __name__ == "__main__":
    main()
