
from pathlib import Path
import pandas as pd
import glob
import random


# -----------------------------
# Configuration
# -----------------------------

RANDOM_SEED = 42

PARQUET_FILES = glob.glob("data/raw/IEMOCAP/*.parquet")
AUDIO_DIR = "data/raw/IEMOCAP/audio"
OUTPUT_DIR = Path("data/metadata")

random.seed(RANDOM_SEED)


# -----------------------------
# Load IEMOCAP metadata
# -----------------------------

from src.datasets.metadata import load_iemocap_metadata

df = load_iemocap_metadata(
    PARQUET_FILES,
    AUDIO_DIR
)

print("Total records:", len(df))


# -----------------------------
# Extract speaker ID
# Example:
# Ses01F_impro01_F000.wav
# Speaker ID: Ses01F
# -----------------------------

df["speaker_id"] = df["utterance_id"].str.extract(
    r"(Ses\d{2}[FM])"
)


# Check missing speaker IDs
missing_speakers = df["speaker_id"].isna().sum()

if missing_speakers > 0:
    raise ValueError(
        f"Missing speaker IDs: {missing_speakers}"
    )


# -----------------------------
# Get unique speakers
# -----------------------------

speakers = sorted(df["speaker_id"].unique())

random.shuffle(speakers)

total_speakers = len(speakers)

train_count = int(total_speakers * 0.60)
validation_count = int(total_speakers * 0.20)

train_speakers = speakers[:train_count]

validation_speakers = speakers[
    train_count:train_count + validation_count
]

test_speakers = speakers[
    train_count + validation_count:
]


# -----------------------------
# Create splits
# -----------------------------

train_df = df[
    df["speaker_id"].isin(train_speakers)
].copy()

validation_df = df[
    df["speaker_id"].isin(validation_speakers)
].copy()

test_df = df[
    df["speaker_id"].isin(test_speakers)
].copy()


# -----------------------------
# Create output directory
# -----------------------------

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# -----------------------------
# Save CSV files
# -----------------------------

train_df.to_csv(
    OUTPUT_DIR / "iemocap_train.csv",
    index=False
)

validation_df.to_csv(
    OUTPUT_DIR / "iemocap_validation.csv",
    index=False
)

test_df.to_csv(
    OUTPUT_DIR / "iemocap_test.csv",
    index=False
)


# -----------------------------
# Display results
# -----------------------------

print("\nSplit Results:")
print("Train speakers:", len(train_speakers))
print("Train records:", len(train_df))

print(
    "Validation speakers:",
    len(validation_speakers)
)

print(
    "Validation records:",
    len(validation_df)
)

print("Test speakers:", len(test_speakers))
print("Test records:", len(test_df))


# -----------------------------
# Check speaker overlap
# -----------------------------

train_set = set(train_df["speaker_id"])
validation_set = set(validation_df["speaker_id"])
test_set = set(test_df["speaker_id"])

print("\nSpeaker Overlap Checks:")

print(
    "Train-Validation:",
    train_set.intersection(validation_set)
)

print(
    "Train-Test:",
    train_set.intersection(test_set)
)

print(
    "Validation-Test:",
    validation_set.intersection(test_set)
)


# -----------------------------
# Check total records
# -----------------------------

total_split_records = (
    len(train_df)
    + len(validation_df)
    + len(test_df)
)

print("\nTotal split records:", total_split_records)

if total_split_records == len(df):
    print("All records included: PASS")
else:
    print("All records included: FAIL")


print("\nIEMOCAP splitting completed!")