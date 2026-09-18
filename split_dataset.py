
import random
from pathlib import Path

from src.datasets.metadata import load_ravdess_metadata


# ============================================
# 1. Load RAVDESS metadata
# ============================================

df = load_ravdess_metadata("data/raw/RAVDESS")

print("Total records:", len(df))

print("\nEmotion counts:")
print(df["emotion"].value_counts())

print("\nUnique speakers:", df["speaker_id"].nunique())

print("\nFirst 5 records:")
print(df.head())


# ============================================
# 2. Get unique speakers
# ============================================

speakers = sorted(df["speaker_id"].unique())

random.seed(42)

random.shuffle(speakers)

n_speakers = len(speakers)


# ============================================
# 3. Split speakers into train, validation, test
# ============================================

train_end = int(0.6 * n_speakers)

val_end = int(0.8 * n_speakers)

train_speakers = speakers[:train_end]

val_speakers = speakers[train_end:val_end]

test_speakers = speakers[val_end:]


# ============================================
# 4. Create datasets using speaker IDs
# ============================================

train_df = df[df["speaker_id"].isin(train_speakers)]

val_df = df[df["speaker_id"].isin(val_speakers)]

test_df = df[df["speaker_id"].isin(test_speakers)]


# ============================================
# 5. Display split information
# ============================================

print("\nTotal speakers:", n_speakers)

print("\nTrain speakers:", len(train_speakers))

print("Validation speakers:", len(val_speakers))

print("Test speakers:", len(test_speakers))

print("\nTrain records:", len(train_df))

print("Validation records:", len(val_df))

print("Test records:", len(test_df))


# ============================================
# 6. Check speaker overlap
# ============================================

train_set = set(train_df["speaker_id"])

val_set = set(val_df["speaker_id"])

test_set = set(test_df["speaker_id"])


print("\nSpeaker overlap checks:")

print(
    "Train and validation overlap:",
    train_set & val_set
)

print(
    "Train and test overlap:",
    train_set & test_set
)

print(
    "Validation and test overlap:",
    val_set & test_set
)


# ============================================
# 7. Check whether all records are included
# ============================================

total_split_records = (
    len(train_df)
    + len(val_df)
    + len(test_df)
)

print("\nTotal split records:", total_split_records)

if total_split_records == len(df):
    print("All records included: True")
else:
    print("All records included: False")


# ============================================
# 8. Display emotion distribution
# ============================================

print("\nTrain emotion distribution:")

print(train_df["emotion"].value_counts())


print("\nValidation emotion distribution:")

print(val_df["emotion"].value_counts())


print("\nTest emotion distribution:")

print(test_df["emotion"].value_counts())


# ============================================
# 9. Save metadata CSV files
# ============================================

output_dir = Path("data/metadata")

output_dir.mkdir(
    parents=True,
    exist_ok=True
)


train_df.to_csv(
    output_dir / "train.csv",
    index=False
)

val_df.to_csv(
    output_dir / "validation.csv",
    index=False
)

test_df.to_csv(
    output_dir / "test.csv",
    index=False
)


# ============================================
# 10. Confirm that files were saved
# ============================================

print("\nMetadata files saved successfully:")

print(output_dir / "train.csv")

print(output_dir / "validation.csv")

print(output_dir / "test.csv")