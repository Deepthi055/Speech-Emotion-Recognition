
from pathlib import Path
import pandas as pd


# Metadata files
metadata_files = {
    "Train": "data/metadata/train.csv",
    "Validation": "data/metadata/validation.csv",
    "Test": "data/metadata/test.csv",
}


# Check each dataset
for dataset_name, file_path in metadata_files.items():

    print(f"\nChecking {dataset_name} dataset...")

    # Load CSV
    df = pd.read_csv(file_path)

    # Count total records
    total_records = len(df)

    # Check whether audio files exist
    existing_files = 0
    missing_files = []

    for audio_path in df["audio_path"]:

        if Path(audio_path).exists():
            existing_files += 1
        else:
            missing_files.append(audio_path)

    missing_count = len(missing_files)

    # Display results
    print("Total records:", total_records)

    print("Existing audio files:", existing_files)

    print("Missing audio files:", missing_count)

    if missing_count == 0:
        print("Status: PASS - All audio files exist")
    else:
        print("Status: FAIL - Some audio files are missing")

        print("\nFirst 5 missing files:")

        for missing_file in missing_files[:5]:
            print(missing_file)