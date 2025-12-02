import argparse
import json
import pathlib
from dataclasses import dataclass
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import accuracy_score, classification_report, f1_score
from torch.utils.data import Dataset
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    Trainer,
    TrainingArguments,
)


PROJECT_ROOT = pathlib.Path(__file__).resolve().parents[1]
PROC = PROJECT_ROOT / "data_proc"
MODELS_DIR = PROJECT_ROOT / "models"
MODELS_DIR.mkdir(exist_ok=True)


class EmotionDataset(Dataset):
    def __init__(self, texts: List[str], labels: List[int], tokenizer, max_length: int = 128):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self) -> int:
        return len(self.texts)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        text = str(self.texts[idx])
        encoding = self.tokenizer(
            text,
            truncation=True,
            padding=False,
            max_length=self.max_length,
        )
        item = {k: torch.tensor(v) for k, v in encoding.items()}
        item["labels"] = torch.tensor(int(self.labels[idx]), dtype=torch.long)
        return item


@dataclass
class EmotionConfig:
    model_name: str
    max_length: int = 128
    num_labels: int = 7


def load_splits() -> Dict[str, pd.DataFrame]:
    train_df = pd.read_csv(PROC / "train.csv")
    val_df = pd.read_csv(PROC / "val.csv")
    test_df = pd.read_csv(PROC / "test.csv")
    return {"train": train_df, "val": val_df, "test": test_df}


def load_label_mapping() -> Dict[int, str]:
    with open(PROC / "emotion_label_map.json", "r") as f:
        emo_id2name = json.load(f)
    # keys are strings in JSON, convert to int
    emo_id2name = {int(k): v for k, v in emo_id2name.items()}
    return emo_id2name


def compute_metrics_builder(id2label: Dict[int, str]):
    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        preds = np.argmax(logits, axis=-1)
        acc = accuracy_score(labels, preds)
        macro_f1 = f1_score(labels, preds, average="macro")
        return {"accuracy": acc, "macro_f1": macro_f1}

    return compute_metrics


def train_and_evaluate(config: EmotionConfig, output_name: str) -> None:
    emo_id2name = load_label_mapping()

    tokenizer = AutoTokenizer.from_pretrained(config.model_name)

    splits = load_splits()
    train_df = splits["train"]
    val_df = splits["val"]
    test_df = splits["test"]

    # Use cleaned text if available, otherwise fall back to raw utterance
    text_col = "cleaned_utterance" if "cleaned_utterance" in train_df.columns else "utterance"

    train_dataset = EmotionDataset(
        texts=train_df[text_col].tolist(),
        labels=train_df["emotion_id"].tolist(),
        tokenizer=tokenizer,
        max_length=config.max_length,
    )
    val_dataset = EmotionDataset(
        texts=val_df[text_col].tolist(),
        labels=val_df["emotion_id"].tolist(),
        tokenizer=tokenizer,
        max_length=config.max_length,
    )
    test_dataset = EmotionDataset(
        texts=test_df[text_col].tolist(),
        labels=test_df["emotion_id"].tolist(),
        tokenizer=tokenizer,
        max_length=config.max_length,
    )

    id2label = {i: emo_id2name[i] for i in sorted(emo_id2name.keys())}
    label2id = {v: k for k, v in id2label.items()}

    model = AutoModelForSequenceClassification.from_pretrained(
        config.model_name,
        num_labels=config.num_labels,
        id2label=id2label,
        label2id=label2id,
    )

    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    run_name = output_name
    out_dir = MODELS_DIR / output_name

    training_args = TrainingArguments(
        output_dir=str(out_dir),
        evaluation_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="macro_f1",
        greater_is_better=True,
        num_train_epochs=3,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=32,
        learning_rate=2e-5,
        weight_decay=0.01,
        logging_steps=50,
        save_total_limit=2,
        report_to=[],  # disable wandb etc.
        run_name=run_name,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics_builder(id2label),
    )

    trainer.train()

    # Evaluate on test set
    preds_output = trainer.predict(test_dataset)
    logits = preds_output.predictions
    labels = preds_output.label_ids
    preds = np.argmax(logits, axis=-1)

    report = classification_report(
        labels,
        preds,
        target_names=[id2label[i] for i in sorted(id2label.keys())],
        digits=4,
    )

    acc = accuracy_score(labels, preds)
    macro_f1 = f1_score(labels, preds, average="macro")

    report_path = MODELS_DIR / f"{output_name}_evaluation_report.txt"
    with open(report_path, "w") as f:
        f.write(f"Model: {config.model_name} fine-tuned on balanced augmented dataset\n")
        f.write("=" * 72 + "\n\n")
        f.write(report)
        f.write("\n\n")
        f.write(f"Accuracy: {acc:.4f}\n")
        f.write(f"Macro F1: {macro_f1:.4f}\n")

    print(f"Saved evaluation report to: {report_path}")


def main(argv: Optional[List[str]] = None) -> None:
    parser = argparse.ArgumentParser(description="Fine-tune transformers for emotion classification")
    parser.add_argument(
        "--model-name",
        type=str,
        required=True,
        help="Hugging Face model name, e.g. 'roberta-base' or 'distilroberta-base'",
    )
    parser.add_argument(
        "--max-length",
        type=int,
        default=128,
        help="Maximum sequence length for tokenization",
    )
    args = parser.parse_args(argv)

    config = EmotionConfig(model_name=args.model_name, max_length=args.max_length)
    safe_name = args.model_name.replace("/", "-")
    output_name = f"{safe_name}_emotion"
    train_and_evaluate(config, output_name)


if __name__ == "__main__":
    main()
