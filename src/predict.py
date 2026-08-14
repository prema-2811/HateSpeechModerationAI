import os
import sys
import re
import torch

from transformers import AutoTokenizer, AutoModelForSequenceClassification

# Allow imports from project root
sys.path.append(
    os.path.dirname(
        os.path.dirname(
            os.path.abspath(__file__)
        )
    )
)

import config

from src.moderation_agent import moderation_decision


# Final selected model
MODEL_PATH = "prema-2811/hatespeech-roberta"

device = torch.device("cpu")


def clean_text(text):

    text = str(text)

    # Remove URLs
    text = re.sub(r"http\S+|www\S+", " ", text)

    # Replace usernames
    text = re.sub(r"@\w+", "@user", text)

    # Remove extra spaces
    text = re.sub(r"\s+", " ", text)

    return text.strip()


print("Loading final RoBERTa model...")

tokenizer = AutoTokenizer.from_pretrained(
    MODEL_PATH
)

model = AutoModelForSequenceClassification.from_pretrained(
    MODEL_PATH
)

model.to(device)
model.eval()

print("Model loaded successfully.\n")


def predict_text(text):

    cleaned_text = clean_text(text)

    encoding = tokenizer(
        cleaned_text,
        truncation=True,
        padding="max_length",
        max_length=config.MAX_LENGTH,
        return_tensors="pt"
    )

    inputs = {
        key: value.to(device)
        for key, value in encoding.items()
    }

    with torch.no_grad():

        outputs = model(**inputs)

        probabilities = torch.softmax(
            outputs.logits,
            dim=1
        )

    confidence, predicted_class = torch.max(
        probabilities,
        dim=1
    )

    predicted_class = predicted_class.item()
    confidence = confidence.item()

    label = config.LABEL_NAMES[
        predicted_class
    ]

    decision = moderation_decision(
        label,
        confidence
    )

    return {
        "label": label,
        "confidence": confidence,
        "action": decision["action"],
        "message": decision["message"]
    }


if __name__ == "__main__":

    print("====================================")
    print("AI HATE SPEECH MODERATION SYSTEM")
    print("====================================")

    text = input(
        "\nEnter social media text: "
    )

    result = predict_text(text)

    print("\n====================================")
    print("PREDICTION RESULT")
    print("====================================")

    print(
        "Prediction:",
        result["label"]
    )

    print(
        "Confidence:",
        f"{result['confidence'] * 100:.2f}%"
    )

    print(
        "Moderation Action:",
        result["action"]
    )

    print(
        "Message:",
        result["message"]
    )