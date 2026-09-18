
from pathlib import Path
import pandas as pd
import random

# Dataset path
DATASET_DIR = Path("data/raw/CREMA-D/AudioWAV")

# Output directory
OUTPUT_DIR = Path("data/metadata")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Emotion mapping
EMOTION_MAP = {
    "ANG": "angry",
    "DIS": "disgust",
    "FEA": "fear",
    "HAP": "happy",
    "NEU": "neutral",
    "SAD": "sad",
}

# Find all WAV files
audio_files = sorted(DATASET_DIR.glob("*.wav"))

if not audio_files:
    raise FileNotFoundError(
        "No WAV files found in the CREMA-D AudioWAV folder."
    )

records = []

for audio_file in audio_files:
    parts = audio_file.stem.split("_")

    # Expected format: Speaker_Sentence_Emotion_Intensity
    if len(parts) != 4:
        print(f"Skipping unexpected filename: {audio_file.name}")
        continue

    speaker_id = parts[0]
    emotion_code = parts[2]

    if emotion_code not in EMOTION_MAP:
        print(f"Skipping unknown emotion: {audio_file.name}")
        continue

    records.append({
        "path": str(audio_file),
        "speaker_id": speaker_id,
        "emotion": EMOTION_MAP[emotion_code],
    })

metadata = pd.DataFrame(records)

if metadata.empty:
    raise ValueError("No valid CREMA-D metadata records were created.")

# Get unique speakers
speakers = sorted(metadata["speaker_id"].unique())

# Reproducible shuffle
random.seed(42)
random.shuffle(speakers)

# 60% train, 20% validation, 20% test
total_speakers = len(speakers)

train_count = int(total_speakers * 0.60)
validation_count = int(total_speakers * 0.20)

train_speakers = set(speakers[:train_count])
validation_speakers = set(
    speakers[train_count:train_count + validation_count]
)
test_speakers = set(
    speakers[train_count + validation_count:]
)

# Create splits
train_df = metadata[
    metadata["speaker_id"].isin(train_speakers)
].copy()

validation_df = metadata[
    metadata["speaker_id"].isin(validation_speakers)
].copy()

test_df = metadata[
    metadata["speaker_id"].isin(test_speakers)
].copy()

# Save CSV files
train_df.to_csv(OUTPUT_DIR / "cremad_train.csv", index=False)
validation_df.to_csv(
    OUTPUT_DIR / "cremad_validation.csv",
    index=False
)
test_df.to_csv(OUTPUT_DIR / "cremad_test.csv", index=False)

# Check speaker overlap
train_val_overlap = train_speakers & validation_speakers
train_test_overlap = train_speakers & test_speakers
validation_test_overlap = validation_speakers & test_speakers

print("\nCREMA-D SPLIT SUMMARY")
print("-" * 40)

print(f"Total audio files: {len(metadata)}")
print(f"Total speakers: {total_speakers}")

print(f"\nTrain speakers: {len(train_speakers)}")
print(f"Train records: {len(train_df)}")

print(f"\nValidation speakers: {len(validation_speakers)}")
print(f"Validation records: {len(validation_df)}")

print(f"\nTest speakers: {len(test_speakers)}")
print(f"Test records: {len(test_df)}")

print("\nSpeaker overlap checks:")
print(f"Train-Validation: {train_val_overlap}")
print(f"Train-Test: {train_test_overlap}")
print(f"Validation-Test: {validation_test_overlap}")

print("\nEmotion distribution:")
print("\nTrain:")
print(train_df["emotion"].value_counts())

print("\nValidation:")
print(validation_df["emotion"].value_counts())

print("\nTest:")
print(test_df["emotion"].value_counts())

print("\nCSV files saved successfully!")