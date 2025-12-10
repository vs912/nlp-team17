"""
Simple Emotion Heartbeat Map Visualization
Visualizes emotion labels across dialogue utterances as a line plot.
Uses ground truth labels from DailyDialog to demonstrate the visualization.
"""

import pathlib
from typing import List

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

PROJECT_ROOT = pathlib.Path(__file__).resolve().parent
PROC = PROJECT_ROOT / "data_proc"

# Emotion labels
EMOTION_LABELS = [
    "anger",
    "disgust",
    "fear",
    "happiness",
    "no_emotion",
    "sadness",
    "surprise",
]


def load_dialogues():
    """Load dialogues from DailyDialog."""
    dd_df = pd.read_csv(PROC / "dailydialog_utterances.csv")
    
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
        markersize=10, 
        linewidth=3, 
        color='#2E86AB',
        label='Ground Truth',
        alpha=0.9,
        zorder=3,
    )
    
    # Plot predicted emotions
    ax.plot(
        x, pred_y, 
        marker='s', 
        markersize=8, 
        linewidth=2.5, 
        color='#A23B72',
        linestyle='--',
        label='Model Prediction',
        alpha=0.8,
        zorder=2,
    )
    
    # Set y-axis labels to emotion names
    ax.set_yticks(range(len(emotion_names)))
    ax.set_yticklabels(emotion_names, fontsize=12, fontweight='bold')
    
    # Set x-axis
    ax.set_xticks(x)
    ax.set_xticklabels([f"ut{i}" for i in x], fontsize=11)
    ax.set_xlabel("Utterance Position", fontsize=14, fontweight='bold')
    ax.set_ylabel("Emotion", fontsize=14, fontweight='bold')
    
    # Title
    ax.set_title(
        f"Emotion Heartbeat Map - Dialogue {dialogue_id}",
        fontsize=16,
        fontweight='bold',
        pad=20,
    )
    
    # Grid
    ax.grid(True, alpha=0.3, linestyle=':', linewidth=1, axis='both')
    ax.set_axisbelow(True)
    
    # Add horizontal lines at each emotion level
    for i in range(len(emotion_names)):
        ax.axhline(y=i, color='gray', linestyle='-', linewidth=0.5, alpha=0.2)
    
    # Legend
    ax.legend(loc='upper right', fontsize=12, framealpha=0.95, shadow=True)
    
    # Set y-axis limits with some padding
    ax.set_ylim(-0.5, len(emotion_names) - 0.5)
    
    # Add utterance text as annotations (for first few)
    max_annotations = min(len(utterances), len(x))
    for i in range(max_annotations):
        # Truncate long utterances
        text = utterances[i][:35] + "..." if len(utterances[i]) > 35 else utterances[i]
        
        # Alternate annotation positions to avoid overlap
        y_offset = 15 if i % 2 == 0 else -15
        va = 'bottom' if i % 2 == 0 else 'top'
        
        ax.annotate(
            text,
            xy=(x[i], true_y[i]),
            xytext=(0, y_offset),
            textcoords='offset points',
            fontsize=8,
            alpha=0.7,
            ha='center',
            va=va,
            bbox=dict(boxstyle='round,pad=0.4', facecolor='lightyellow', alpha=0.6, edgecolor='gray'),
            arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0', color='gray', alpha=0.4),
        )
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=200, bbox_inches='tight')
    plt.close()
    
    print(f"  ✓ Saved: {save_path.name}")


def simulate_predictions(true_labels: List[int], accuracy: float = 0.75) -> List[int]:
    """
    Simulate model predictions with a given accuracy.
    For demonstration purposes when actual model is not available.
    """
    np.random.seed(42)
    pred_labels = []
    
    for true_label in true_labels:
        if np.random.random() < accuracy:
            # Correct prediction
            pred_labels.append(true_label)
        else:
            # Random incorrect prediction
            wrong_labels = [l for l in range(7) if l != true_label]
            pred_labels.append(np.random.choice(wrong_labels))
    
    return pred_labels


def main():
    print("=" * 70)
    print("Emotion Heartbeat Map Visualization")
    print("=" * 70)
    
    # Load dialogues
    print("\nLoading dialogues from DailyDialog...")
    dialogues, labels, dialog_ids = load_dialogues()
    print(f"✓ Loaded {len(dialogues)} dialogues")
    
    # Create output directory
    viz_dir = PROJECT_ROOT / "visualizations"
    viz_dir.mkdir(exist_ok=True)
    
    # Select interesting dialogues to visualize
    print("\nSelecting dialogues for visualization...")
    
    selected_indices = []
    for i, (dialog, labs) in enumerate(zip(dialogues, labels)):
        # Select dialogues with 4-10 utterances and at least 3 different emotions
        if 4 <= len(dialog) <= 10 and len(set(labs)) >= 3:
            selected_indices.append(i)
            if len(selected_indices) >= 8:  # Visualize 8 dialogues
                break
    
    if not selected_indices:
        # Fallback: just take first 8 dialogues with reasonable length
        selected_indices = [i for i in range(len(dialogues)) if 4 <= len(dialogues[i]) <= 10][:8]
    
    print(f"✓ Selected {len(selected_indices)} dialogues for visualization")
    
    # Generate visualizations
    print("\nGenerating emotion heartbeat maps...")
    print("-" * 70)
    
    for idx in selected_indices:
        utterances = dialogues[idx]
        true_labs = labels[idx]
        actual_dialog_id = dialog_ids[idx]
        
        print(f"\nDialogue {actual_dialog_id}:")
        print(f"  Utterances: {len(utterances)}")
        print(f"  Emotions: {[EMOTION_LABELS[l] for l in true_labs]}")
        
        # Simulate predictions (in real scenario, use actual model)
        pred_labs = simulate_predictions(true_labs, accuracy=0.70)
        
        # Plot
        save_path = viz_dir / f"emotion_heartbeat_dialogue_{actual_dialog_id}.png"
        plot_emotion_heartbeat(
            utterances=utterances,
            true_labels=true_labs,
            pred_labels=pred_labs,
            emotion_names=EMOTION_LABELS,
            dialogue_id=str(actual_dialog_id),
            save_path=save_path,
        )
    
    print("\n" + "=" * 70)
    print(f"✓ All visualizations saved to: {viz_dir}/")
    print("=" * 70)
    print("\nYou can now view the emotion heartbeat maps!")
    print(f"Open the '{viz_dir.name}' folder to see the generated plots.")


if __name__ == "__main__":
    main()
