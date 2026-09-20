import sys
from pathlib import Path
import random
import pandas as pd
import soundfile as sf

# Ensure src module is in path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.datasets.metadata import (
    load_cremad_metadata,
    load_ravdess_metadata,
    load_iemocap_metadata,
    extract_iemocap_audio,
    UNIFIED_COLUMNS,
)

def create_speaker_splits(df: pd.DataFrame, train_ratio: float = 0.70, val_ratio: float = 0.15, seed: int = 42) -> pd.DataFrame:
    """Split dataframe into train/val/test based on speaker IDs (no speaker overlap)."""
    random.seed(seed)
    df = df.copy()

    speakers = sorted(df["speaker_id"].unique())
    random.shuffle(speakers)

    total_spks = len(speakers)
    train_count = max(1, int(total_spks * train_ratio))
    val_count = max(1, int(total_spks * val_ratio))

    train_spks = set(speakers[:train_count])
    val_spks = set(speakers[train_count:train_count + val_count])
    test_spks = set(speakers[train_count + val_count:])

    # Handle edge case where test_spks is empty
    if not test_spks:
        test_spks = val_spks

    def assign_split(spk):
        if spk in train_spks:
            return "train"
        elif spk in val_spks:
            return "val"
        else:
            return "test"

    df["split"] = df["speaker_id"].apply(assign_split)
    return df

def validate_dataset(df: pd.DataFrame) -> tuple[int, int]:
    """Check that all audio files exist and check audio format."""
    missing = 0
    valid = 0

    for idx, row in df.iterrows():
        fpath = Path(row["file_path"])
        if not fpath.exists():
            missing += 1
            print(f"[MISSING] {fpath}")
        else:
            valid += 1

    return valid, missing

def main():
    project_root = Path(__file__).resolve().parent.parent
    data_dir = project_root / "data"
    raw_dir = data_dir / "raw"
    metadata_dir = data_dir / "metadata"
    metadata_dir.mkdir(parents=True, exist_ok=True)

    print("=== Step 1: Processing IEMOCAP Audio & Metadata ===")
    iemocap_dir = raw_dir / "IEMOCAP"
    iemocap_parquet_files = sorted(list(iemocap_dir.glob("*.parquet")))
    iemocap_wav_dir = iemocap_dir / "AudioWAV"

    if iemocap_parquet_files:
        print(f"Extracting IEMOCAP audio bytes to {iemocap_wav_dir}...")
        extract_iemocap_audio(iemocap_parquet_files, iemocap_wav_dir)
        df_iemocap = load_iemocap_metadata(iemocap_parquet_files, iemocap_wav_dir)
    else:
        print("WARNING: No IEMOCAP parquet files found.")
        df_iemocap = pd.DataFrame(columns=UNIFIED_COLUMNS)

    print(f"IEMOCAP loaded: {len(df_iemocap)} samples across {df_iemocap['speaker_id'].nunique()} speakers.")

    print("\n=== Step 2: Processing CREMA-D Metadata ===")
    cremad_dir = raw_dir / "CREMA-D"
    df_cremad = load_cremad_metadata(cremad_dir)
    print(f"CREMA-D loaded: {len(df_cremad)} samples across {df_cremad['speaker_id'].nunique()} speakers.")

    print("\n=== Step 3: Processing RAVDESS Metadata ===")
    ravdess_dir = raw_dir / "RAVDESS"
    df_ravdess = load_ravdess_metadata(ravdess_dir)
    print(f"RAVDESS loaded: {len(df_ravdess)} samples across {df_ravdess['speaker_id'].nunique()} speakers.")

    print("\n=== Step 4: Creating Speaker-Independent Splits per Corpus ===")
    df_iemocap = create_speaker_splits(df_iemocap, train_ratio=0.70, val_ratio=0.15, seed=42)
    df_cremad = create_speaker_splits(df_cremad, train_ratio=0.70, val_ratio=0.15, seed=42)
    df_ravdess = create_speaker_splits(df_ravdess, train_ratio=0.70, val_ratio=0.15, seed=42)

    df_full = pd.concat([df_iemocap, df_cremad, df_ravdess], ignore_index=True)

    print("\n=== Step 5: Validating Audio Files ===")
    valid_count, missing_count = validate_dataset(df_full)
    print(f"Validation summary: {valid_count} valid files, {missing_count} missing files.")

    print("\n=== Step 6: Verifying Speaker Isolation Across Splits ===")
    for corpus_name, group in df_full.groupby("corpus"):
        train_spks = set(group[group["split"] == "train"]["speaker_id"])
        val_spks = set(group[group["split"] == "val"]["speaker_id"])
        test_spks = set(group[group["split"] == "test"]["speaker_id"])

        train_val_overlap = train_spks.intersection(val_spks)
        train_test_overlap = train_spks.intersection(test_spks)
        val_test_overlap = val_spks.intersection(test_spks)

        print(f"Corpus: {corpus_name}")
        print(f"  Speakers -> Train: {len(train_spks)}, Val: {len(val_spks)}, Test: {len(test_spks)}")
        print(f"  Overlap  -> Train-Val: {len(train_val_overlap)}, Train-Test: {len(train_test_overlap)}, Val-Test: {len(val_test_overlap)}")
        assert len(train_val_overlap) == 0 and len(train_test_overlap) == 0 and len(val_test_overlap) == 0, f"Speaker overlap detected in {corpus_name}!"

    print("\n=== Step 7: Saving Unified Metadata CSVs ===")
    df_full.to_csv(metadata_dir / "unified_full.csv", index=False)

    df_train = df_full[df_full["split"] == "train"].reset_index(drop=True)
    df_val = df_full[df_full["split"] == "val"].reset_index(drop=True)
    df_test = df_full[df_full["split"] == "test"].reset_index(drop=True)

    df_train.to_csv(metadata_dir / "unified_train.csv", index=False)
    df_val.to_csv(metadata_dir / "unified_val.csv", index=False)
    df_test.to_csv(metadata_dir / "unified_test.csv", index=False)

    print("\nData Split Summary:")
    print(f"  Train samples: {len(df_train)}")
    print(f"  Val samples  : {len(df_val)}")
    print(f"  Test samples : {len(df_test)}")
    print(f"  Total samples: {len(df_full)}")

    print("\nPer-Corpus Split Breakdown:")
    print(pd.crosstab(df_full["corpus"], df_full["split"]))

    print("\nPer-Emotion Split Breakdown:")
    print(pd.crosstab(df_full["emotion"], df_full["split"]))

    print("\nDataset preparation completed successfully.")

if __name__ == "__main__":
    main()
