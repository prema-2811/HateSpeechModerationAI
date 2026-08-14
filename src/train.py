import os
import sys
import random

import numpy as np
import pandas as pd
import torch

from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from torch.nn import CrossEntropyLoss

from transformers import AutoTokenizer, AutoModelForSequenceClassification
from sklearn.metrics import accuracy_score, f1_score

# Allow importing config.py from project root
sys.path.append(
    os.path.dirname(
        os.path.dirname(
            os.path.abspath(__file__)
        )
    )
)

import config


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


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


def calculate_class_weights(train_df):

    counts = train_df["class"].value_counts().sort_index()

    total_samples = len(train_df)
    number_of_classes = len(counts)

    weights = []

    for class_id in range(number_of_classes):

        class_count = counts[class_id]

        weight = total_samples / (
            number_of_classes * class_count
        )

        weights.append(weight)

    return torch.tensor(
        weights,
        dtype=torch.float32
    )


def evaluate(model, dataloader, device):

    model.eval()

    predictions = []
    true_labels = []

    with torch.no_grad():

        for batch in dataloader:

            labels = batch.pop("labels").to(device)

            inputs = {
                key: value.to(device)
                for key, value in batch.items()
            }

            outputs = model(**inputs)

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

    macro_f1 = f1_score(
        true_labels,
        predictions,
        average="macro"
    )

    return accuracy, macro_f1


def train_model():

    set_seed(config.RANDOM_SEED)

    device = torch.device("cpu")

    print("\n========================================")
    print("ROBERTA HATE SPEECH MODEL TRAINING")
    print("========================================")

    print("\nDevice:", device)
    print("Model:", config.MODEL_NAME)

    print("\nLoading datasets...")

    train_df = pd.read_csv(
        config.TRAIN_FILE
    )

    val_df = pd.read_csv(
        config.VAL_FILE
    )

    print(
        "Training samples:",
        len(train_df)
    )

    print(
        "Validation samples:",
        len(val_df)
    )

    print("\nLoading tokenizer...")

    tokenizer = AutoTokenizer.from_pretrained(
        config.MODEL_NAME
    )

    print("Loading RoBERTa model...")

    model = AutoModelForSequenceClassification.from_pretrained(
        config.MODEL_NAME,
        num_labels=config.NUM_LABELS
    )

    model.to(device)
    model.float()

    train_dataset = HateSpeechDataset(
        train_df,
        tokenizer,
        config.MAX_LENGTH
    )

    val_dataset = HateSpeechDataset(
        val_df,
        tokenizer,
        config.MAX_LENGTH
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=True
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=False
    )

    class_weights = calculate_class_weights(
        train_df
    ).to(device)

    print("\nClass weights:")
    print(class_weights)

    loss_function = CrossEntropyLoss(
        weight=class_weights
    )

    optimizer = AdamW(
        model.parameters(),
        lr=config.LEARNING_RATE
    )

    best_f1 = 0.0

    output_folder = "models/roberta"

    os.makedirs(
        output_folder,
        exist_ok=True
    )

    for epoch in range(config.EPOCHS):

        print("\n========================================")
        print(
            f"Epoch {epoch + 1}/{config.EPOCHS}"
        )
        print("========================================")

        model.train()

        total_loss = 0

        for batch_number, batch in enumerate(
            train_loader
        ):

            labels = batch.pop(
                "labels"
            ).to(device)

            inputs = {
                key: value.to(device)
                for key, value in batch.items()
            }

            optimizer.zero_grad()

            outputs = model(**inputs)

            loss = loss_function(
                outputs.logits,
                labels
            )

            loss.backward()

            optimizer.step()

            total_loss += loss.item()

            if (batch_number + 1) % 100 == 0:

                print(
                    f"Batch {batch_number + 1}/"
                    f"{len(train_loader)}"
                    f" | Loss: {loss.item():.4f}"
                )

        average_loss = (
            total_loss / len(train_loader)
        )

        print(
            f"\nAverage training loss: "
            f"{average_loss:.4f}"
        )

        print("\nEvaluating validation set...")

        val_accuracy, val_f1 = evaluate(
            model,
            val_loader,
            device
        )

        print(
            f"Validation Accuracy: "
            f"{val_accuracy * 100:.2f}%"
        )

        print(
            f"Validation Macro F1: "
            f"{val_f1:.4f}"
        )

        if val_f1 > best_f1:

            best_f1 = val_f1

            print("\nNew best RoBERTa model found!")

            model.save_pretrained(
                output_folder
            )

            tokenizer.save_pretrained(
                output_folder
            )

            print(
                "Model saved to:",
                output_folder
            )

    print("\n========================================")
    print("TRAINING COMPLETED")
    print("========================================")

    print(
        "Best Validation Macro F1:",
        round(best_f1, 4)
    )


if __name__ == "__main__":
    train_model()