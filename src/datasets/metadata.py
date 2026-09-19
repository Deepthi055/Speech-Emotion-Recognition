import os
from pathlib import Path
import pandas as pd

EMOTION_LABELS = [
    "angry",
    "disgust",
    "fear",
    "happy",
    "neutral",
    "sad",
]

UNIFIED_COLUMNS = [
    "file_path",
    "emotion",
    "speaker_id",
    "corpus",
    "split",
]

def load_ravdess_metadata(root_dir: str | Path) -> pd.DataFrame:
    """Load and normalize RAVDESS metadata."""
    root = Path(root_dir)

    emotion_map = {
        "01": "neutral",
        "03": "happy",
        "04": "sad",
        "05": "angry",
        "06": "fear",
        "07": "disgust",
    }

    rows = []
    seen = set()

    for audio_path in root.rglob("*.wav"):
        filename = audio_path.stem
        parts = filename.split("-")

        # RAVDESS filenames must contain 7 fields.
        if len(parts) != 7:
            continue

        emotion_code = parts[2]
        actor_id = parts[6]

        # Exclude emotions outside common label space (02=calm, 08=surprised)
        if emotion_code not in emotion_map:
            continue

        # Deduplicate the flat + nested copies.
        if filename in seen:
            continue
        seen.add(filename)

        rows.append(
            {
                "file_path": str(audio_path),
                "emotion": emotion_map[emotion_code],
                "speaker_id": f"ravdess_{actor_id}",
                "corpus": "RAVDESS",
                "split": "",
            }
        )

    return pd.DataFrame(rows, columns=UNIFIED_COLUMNS)

def extract_iemocap_audio(parquet_files: list[str | Path], output_dir: str | Path) -> None:
    """Extract embedded IEMOCAP WAV bytes into audio files on disk if not already present."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    for pfile in parquet_files:
        df = pd.read_parquet(pfile)
        for _, row in df.iterrows():
            audio_data = row["audio"]
            filename = row["file"]
            output_path = output_dir / filename
            if not output_path.exists():
                with open(output_path, "wb") as f:
                    f.write(audio_data["bytes"])

def load_iemocap_metadata(parquet_files: list[str | Path], audio_dir: str | Path) -> pd.DataFrame:
    """Load and normalize IEMOCAP metadata."""
    emotion_map = {
        "angry": "angry",
        "frustrated": "angry",
        "disgust": "disgust",
        "fear": "fear",
        "happy": "happy",
        "excited": "happy",
        "sad": "sad",
        "neutral": "neutral",
    }

    frames = [pd.read_parquet(file) for file in parquet_files]
    df = pd.concat(frames, ignore_index=True)

    rows = []
    seen = set()

    for _, row in df.iterrows():
        emotion = emotion_map.get(row["major_emotion"])
        if emotion is None:
            continue

        filename = row["file"]
        if filename in seen:
            continue
        seen.add(filename)

        session_gender = filename[:6]  # e.g. Ses01F
        audio_path = Path(audio_dir) / filename

        rows.append(
            {
                "file_path": str(audio_path),
                "emotion": emotion,
                "speaker_id": f"iemocap_{session_gender}",
                "corpus": "IEMOCAP",
                "split": "",
            }
        )

    return pd.DataFrame(rows, columns=UNIFIED_COLUMNS)

def load_cremad_metadata(root_dir: str | Path) -> pd.DataFrame:
    """Load and normalize CREMA-D metadata."""
    root = Path(root_dir)

    emotion_map = {
        "ANG": "angry",
        "DIS": "disgust",
        "FEA": "fear",
        "HAP": "happy",
        "NEU": "neutral",
        "SAD": "sad",
    }

    rows = []
    seen = set()

    for audio_path in root.rglob("*.wav"):
        filename = audio_path.stem
        parts = filename.split("_")

        if len(parts) != 4:
            continue

        actor_id = parts[0]
        emotion_code = parts[2]

        if emotion_code not in emotion_map:
            continue

        if filename in seen:
            continue
        seen.add(filename)

        rows.append(
            {
                "file_path": str(audio_path),
                "emotion": emotion_map[emotion_code],
                "speaker_id": f"cremad_{actor_id}",
                "corpus": "CREMA-D",
                "split": "",
            }
        )

    return pd.DataFrame(rows, columns=UNIFIED_COLUMNS)
