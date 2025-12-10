"""
Test Improved Model on Custom Dialogues
Creates custom test dialogues and visualizes emotion predictions.
"""

import pathlib
from typing import List, Tuple

import torch
import matplotlib.pyplot as plt
from transformers import AutoTokenizer

from src.train_roberta_crf_improved import (
    ImprovedRobertaCRFModel,
    compute_transition_prior,
    compute_class_weights,
    EMOTION_LABELS,
    ID2LABEL,
)

PROJECT_ROOT = pathlib.Path(__file__).resolve().parent
MODELS_DIR = PROJECT_ROOT / "models"


# Custom test dialogues with expected emotions
CUSTOM_DIALOGUES = [
    {
        "id": "customer_service",
        "utterances": [
            "Hi, I ordered a laptop last week but it still hasn't arrived.",
            "I'm so sorry for the inconvenience! Let me check your order status right away.",
            "This is unacceptable! I needed it for an important presentation!",
            "I completely understand your frustration. I'll expedite the shipping immediately at no extra cost.",
            "Oh, thank you so much! That would really help me out.",
        ],
        "expected_emotions": ["sadness", "no_emotion", "anger", "no_emotion", "happiness"],
    },
    {
        "id": "surprise_party",
        "utterances": [
            "Hey, can you come to my place tonight around 7?",
            "Sure, what's the occasion?",
            "SURPRISE! Happy Birthday!",
            "Oh my god! I can't believe you did this for me!",
            "We wanted to make your day special!",
        ],
        "expected_emotions": ["no_emotion", "no_emotion", "happiness", "surprise", "happiness"],
    },
    {
        "id": "bad_news",
        "utterances": [
            "I need to tell you something important.",
            "What is it? You're making me nervous.",
            "I'm afraid I have some bad news about your test results.",
            "Oh no... this can't be happening.",
            "I'm here to support you through this. We'll figure it out together.",
        ],
        "expected_emotions": ["no_emotion", "fear", "sadness", "sadness", "no_emotion"],
    },
    {
        "id": "restaurant_complaint",
        "utterances": [
            "Excuse me, there's a hair in my soup!",
            "I'm terribly sorry sir, that's disgusting. Let me get you a new bowl immediately.",
            "This is completely unacceptable! I want to speak to the manager!",
            "I understand your anger. The manager will be right with you, and your meal is on the house.",
            "Well, I appreciate you taking responsibility. Thank you.",
        ],
        "expected_emotions": ["disgust", "disgust", "anger", "no_emotion", "no_emotion"],
    },
    {
        "id": "job_interview",
        "utterances": [
            "Thank you for coming in today. Tell me about yourself.",
            "I'm really nervous, but excited to be here. I've wanted to work here for years.",
            "That's wonderful to hear! Your resume is very impressive.",
            "Thank you so much! That means a lot to me.",
            "We'd like to offer you the position. Congratulations!",
        ],
        "expected_emotions": ["no_emotion", "fear", "happiness", "happiness", "happiness"],
    },
]


def load_model(model_path: pathlib.Path, device: str):
    """Load the trained improved model."""
    # We need transition prior and class weights (use dummy values for inference)
    num_labels = len(EMOTION_LABELS)
    transition_prior = torch.eye(num_labels)  # Dummy
    class_weights = torch.ones(num_labels)  # Dummy
    
    model = ImprovedRobertaCRFModel(
        model_name="roberta-base",
        num_labels=num_labels,
        transition_prior=transition_prior,
        class_weights=class_weights,
        reg_lambda=0.05,
        dropout=0.3,
    ).to(device)
    
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    
    return model


def predict_dialogue(
    model: ImprovedRobertaCRFModel,
    tokenizer,
    utterances: List[str],
    device: str,
    max_length: int = 128,
) -> List[int]:
    """Predict emotions for a dialogue."""
    model.eval()
    
    # Tokenize
    encodings = tokenizer(
        utterances,
        padding=True,
        truncation=True,
        max_length=max_length,
        return_tensors="pt",
    )
    encodings = {k: v.to(device) for k, v in encodings.items()}
    
    # Add batch dimension
    batch_size = 1
    dialog_len = len(utterances)
    dialog_mask = torch.ones((batch_size, dialog_len), dtype=torch.bool, device=device)
    utt_indices = [(0, t) for t in range(dialog_len)]
    encodings["utt_indices"] = utt_indices
    
    with torch.no_grad():
        predictions = model(encodings, dialog_mask=dialog_mask)
    
    return predictions[0]


def plot_custom_dialogue_heartbeat(
    dialogue_id: str,
    utterances: List[str],
    expected_emotions: List[str],
    predicted_emotions: List[str],
    save_path: pathlib.Path,
):
    """Create heartbeat visualization for custom dialogue."""
    # Convert emotion names to IDs
    emotion_to_id = {name: i for i, name in enumerate(EMOTION_LABELS)}
    expected_ids = [emotion_to_id[e] for e in expected_emotions]
    predicted_ids = [emotion_to_id[e] for e in predicted_emotions]
    
    x = list(range(1, len(utterances) + 1))
    
    # Create figure
    fig, ax = plt.subplots(figsize=(16, 9))
    
    # Plot expected emotions
    ax.plot(
        x, expected_ids,
        marker='o',
        markersize=12,
        linewidth=3.5,
        color='#2E86AB',
        label='Expected Emotion',
        alpha=0.9,
        zorder=3,
    )
    
    # Plot predicted emotions
    ax.plot(
        x, predicted_ids,
        marker='s',
        markersize=10,
        linewidth=3,
        color='#A23B72',
        linestyle='--',
        label='Model Prediction',
        alpha=0.85,
        zorder=2,
    )
    
    # Highlight correct predictions
    for i, (exp, pred) in enumerate(zip(expected_ids, predicted_ids)):
        if exp == pred:
            ax.plot(x[i], pred, marker='*', markersize=20, color='#27AE60', zorder=4)
    
    # Y-axis
    ax.set_yticks(range(len(EMOTION_LABELS)))
    ax.set_yticklabels(EMOTION_LABELS, fontsize=13, fontweight='bold')
    
    # X-axis
    ax.set_xticks(x)
    ax.set_xticklabels([f"Utterance {i}" for i in x], fontsize=12)
    ax.set_xlabel("Dialogue Flow", fontsize=15, fontweight='bold')
    ax.set_ylabel("Emotion", fontsize=15, fontweight='bold')
    
    # Title
    accuracy = sum(e == p for e, p in zip(expected_ids, predicted_ids)) / len(expected_ids) * 100
    ax.set_title(
        f"Emotion Heartbeat Map: {dialogue_id.replace('_', ' ').title()}\n"
        f"Prediction Accuracy: {accuracy:.1f}%",
        fontsize=17,
        fontweight='bold',
        pad=20,
    )
    
    # Grid
    ax.grid(True, alpha=0.3, linestyle=':', linewidth=1)
    ax.set_axisbelow(True)
    
    # Horizontal lines
    for i in range(len(EMOTION_LABELS)):
        ax.axhline(y=i, color='gray', linestyle='-', linewidth=0.5, alpha=0.2)
    
    # Legend
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], marker='o', color='#2E86AB', linewidth=3, markersize=10, label='Expected'),
        Line2D([0], [0], marker='s', color='#A23B72', linewidth=3, linestyle='--', markersize=10, label='Predicted'),
        Line2D([0], [0], marker='*', color='#27AE60', linewidth=0, markersize=15, label='Correct Match'),
    ]
    ax.legend(handles=legend_elements, loc='upper right', fontsize=13, framealpha=0.95, shadow=True)
    
    ax.set_ylim(-0.5, len(EMOTION_LABELS) - 0.5)
    
    # Add utterance text
    for i, utt in enumerate(utterances):
        text = utt[:50] + "..." if len(utt) > 50 else utt
        y_offset = 20 if i % 2 == 0 else -20
        va = 'bottom' if i % 2 == 0 else 'top'
        
        ax.annotate(
            text,
            xy=(x[i], expected_ids[i]),
            xytext=(0, y_offset),
            textcoords='offset points',
            fontsize=9,
            alpha=0.8,
            ha='center',
            va=va,
            bbox=dict(boxstyle='round,pad=0.5', facecolor='lightyellow', alpha=0.7, edgecolor='gray', linewidth=1.5),
            arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0.1', color='gray', alpha=0.5, linewidth=1.5),
        )
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=200, bbox_inches='tight')
    plt.close()
    
    print(f"  ✓ Saved: {save_path.name}")


def main():
    print("=" * 80)
    print("Testing Improved Model on Custom Dialogues")
    print("=" * 80)
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"\nUsing device: {device}")
    
    # Load model
    model_path = MODELS_DIR / "roberta_crf_improved_best.pt"
    
    if not model_path.exists():
        print(f"\n❌ Error: Model not found at {model_path}")
        print("Please train the improved model first:")
        print("  python -m src.train_roberta_crf_improved")
        return
    
    print(f"\nLoading model from {model_path}...")
    model = load_model(model_path, device)
    tokenizer = AutoTokenizer.from_pretrained("roberta-base")
    print("✓ Model loaded successfully")
    
    # Create output directory
    viz_dir = PROJECT_ROOT / "custom_visualizations"
    viz_dir.mkdir(exist_ok=True)
    
    # Test on custom dialogues
    print(f"\nTesting on {len(CUSTOM_DIALOGUES)} custom dialogues...")
    print("=" * 80)
    
    total_correct = 0
    total_predictions = 0
    
    for dialogue_data in CUSTOM_DIALOGUES:
        dialogue_id = dialogue_data["id"]
        utterances = dialogue_data["utterances"]
        expected_emotions = dialogue_data["expected_emotions"]
        
        print(f"\n📝 Dialogue: {dialogue_id.replace('_', ' ').title()}")
        print(f"   Utterances: {len(utterances)}")
        
        # Predict
        pred_ids = predict_dialogue(model, tokenizer, utterances, device)
        predicted_emotions = [EMOTION_LABELS[i] for i in pred_ids]
        
        # Calculate accuracy
        correct = sum(e == p for e, p in zip(expected_emotions, predicted_emotions))
        accuracy = correct / len(expected_emotions) * 100
        
        total_correct += correct
        total_predictions += len(expected_emotions)
        
        print(f"\n   Expected:  {expected_emotions}")
        print(f"   Predicted: {predicted_emotions}")
        print(f"   Accuracy:  {accuracy:.1f}% ({correct}/{len(expected_emotions)} correct)")
        
        # Visualize
        save_path = viz_dir / f"custom_dialogue_{dialogue_id}.png"
        plot_custom_dialogue_heartbeat(
            dialogue_id=dialogue_id,
            utterances=utterances,
            expected_emotions=expected_emotions,
            predicted_emotions=predicted_emotions,
            save_path=save_path,
        )
    
    # Overall statistics
    overall_accuracy = total_correct / total_predictions * 100
    
    print("\n" + "=" * 80)
    print("OVERALL RESULTS")
    print("=" * 80)
    print(f"Total Predictions: {total_predictions}")
    print(f"Correct Predictions: {total_correct}")
    print(f"Overall Accuracy: {overall_accuracy:.2f}%")
    print(f"\n✓ All visualizations saved to: {viz_dir}/")
    print("=" * 80)


if __name__ == "__main__":
    main()
