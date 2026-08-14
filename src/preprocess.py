import pandas as pd
import re
from sklearn.model_selection import train_test_split

# File paths
INPUT_FILE = "data/raw/labeled_data.csv"
TRAIN_FILE = "data/processed/train.csv"
VAL_FILE = "data/processed/validation.csv"
TEST_FILE = "data/processed/test.csv"


def clean_text(text):
    text = str(text)

    # Remove URLs
    text = re.sub(r"http\S+|www\S+", " ", text)

    # Replace Twitter usernames with a generic token
    text = re.sub(r"@\w+", "@user", text)

    # Remove extra spaces
    text = re.sub(r"\s+", " ", text)

    return text.strip()


print("Loading dataset...")

df = pd.read_csv(INPUT_FILE)

# Keep only the columns needed for training
df = df[["tweet", "class"]].copy()

# Remove duplicate tweets
df = df.drop_duplicates(subset="tweet")

# Clean tweet text
df["tweet"] = df["tweet"].apply(clean_text)

# Remove empty tweets if any
df = df[df["tweet"].str.len() > 0]

print("Dataset size after cleaning:", len(df))

print("\nClass distribution:")
print(df["class"].value_counts().sort_index())

# First split:
# 80% training
# 20% temporary
train_df, temp_df = train_test_split(
    df,
    test_size=0.20,
    random_state=42,
    stratify=df["class"]
)

# Split the remaining 20% equally:
# 10% validation
# 10% test
val_df, test_df = train_test_split(
    temp_df,
    test_size=0.50,
    random_state=42,
    stratify=temp_df["class"]
)

# Save files
train_df.to_csv(TRAIN_FILE, index=False)
val_df.to_csv(VAL_FILE, index=False)
test_df.to_csv(TEST_FILE, index=False)

print("\nData split completed.")

print("Training samples:", len(train_df))
print("Validation samples:", len(val_df))
print("Test samples:", len(test_df))

print("\nFiles saved successfully:")
print(TRAIN_FILE)
print(VAL_FILE)
print(TEST_FILE)