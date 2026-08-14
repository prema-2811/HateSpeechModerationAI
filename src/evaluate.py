import os
import sys

import pandas as pd
import torch

from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModelForSequenceClassification

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix
)

sys.path.append(
    os.path.dirname(
        os.path.dirname(
            os.path.abspath(__file__)
        )
    )
)

import config


class HateSpeechDataset(Dataset):

    def __init__(self, dataframe, tokenizer, max_length):

        self.texts = dataframe["tweet"].astype(str).tolist()
        self.labels = dataframe["class"].astype(int).tolist()

        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, index):

        text = self.texts[index]
        label = self.labels[index]

        encoding = self.tokenizer(
            text,
            truncation=True,
            padding="max_length",
            max_length=self.max_length,
            return_tensors="pt"
        )

        item = {
            key: value.squeeze(0)
            for key, value in encoding.items()
        }

        item["labels"] = torch.tensor(
            label,
            dtype=torch.long
        )

        return item


def evaluate_model(model_name):

    device = torch.device("cpu")

    model_path = os.path.join(
        "models",
        model_name
    )

    print("\n====================================")
    print("MODEL EVALUATION")
    print("====================================")

    print("Model:", model_name)
    print("Path:", model_path)

    print("\nLoading test dataset...")

    test_df = pd.read_csv(
        config.TEST_FILE
    )

    print(
        "Test samples:",
        len(test_df)
    )

    print("\nLoading tokenizer and model...")

    tokenizer = AutoTokenizer.from_pretrained(
        model_path
    )

    model = AutoModelForSequenceClassification.from_pretrained(
        model_path
    )

    model.to(device)
    model.eval()

    test_dataset = HateSpeechDataset(
        test_df,
        tokenizer,
        config.MAX_LENGTH
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=False
    )

    predictions = []
    true_labels = []

    print("\nTesting model...")

    with torch.no_grad():

        for batch_number, batch in enumerate(test_loader):

            labels = batch.pop(
                "labels"
            ).to(device)

            inputs = {
                key: value.to(device)
                for key, value in batch.items()
            }

            outputs = model(
                **inputs
            )

            predicted = torch.argmax(
                outputs.logits,
                dim=1
            )

            predictions.extend(
                predicted.cpu().numpy()
            )

            true_labels.extend(
                labels.cpu().numpy()
            )

    accuracy = accuracy_score(
        true_labels,
        predictions
    )

    precision = precision_score(
        true_labels,
        predictions,
        average="macro",
        zero_division=0
    )

    recall = recall_score(
        true_labels,
        predictions,
        average="macro",
        zero_division=0
    )

    macro_f1 = f1_score(
        true_labels,
        predictions,
        average="macro",
        zero_division=0
    )

    print("\n====================================")
    print("TEST RESULTS")
    print("====================================")

    print(
        f"Accuracy: {accuracy * 100:.2f}%"
    )

    print(
        f"Macro Precision: {precision:.4f}"
    )

    print(
        f"Macro Recall: {recall:.4f}"
    )

    print(
        f"Macro F1: {macro_f1:.4f}"
    )

    print("\nClassification Report:\n")

    print(
        classification_report(
            true_labels,
            predictions,
            target_names=[
                "Hate Speech",
                "Offensive Language",
                "Neither"
            ],
            zero_division=0
        )
    )

    print("Confusion Matrix:")

    print(
        confusion_matrix(
            true_labels,
            predictions
        )
    )


if __name__ == "__main__":

    evaluate_model("roberta")