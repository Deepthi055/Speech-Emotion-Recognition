from pathlib import Path
import pandas as pd


EMOTION_LABELS = [
    "angry",
    "disgust",
    "fear",
    "happy",
    "sad",
    "neutral",
]

METADATA_COLUMNS = [
    "utterance_id",
    "corpus",
    "speaker_id",
    "emotion",
    "audio_path",
]
def load_ravdess_metadata(root_dir):
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

        # Exclude emotions that are outside our common label space.
        if emotion_code not in emotion_map:
            continue

        # Deduplicate the flat + nested copies.
        if filename in seen:
            continue
        seen.add(filename)

        rows.append(
            {
                "utterance_id": f"ravdess_{filename}",
                "corpus": "RAVDESS",
                "speaker_id": f"ravdess_{actor_id}",
                "emotion": emotion_map[emotion_code],
                "audio_path": str(audio_path),
            }
        )

    return pd.DataFrame(rows, columns=METADATA_COLUMNS)


def extract_iemocap_audio(df, output_dir):
    """Extract embedded IEMOCAP WAV bytes into audio files."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    extracted_paths = []

    for _, row in df.iterrows():
        audio_data = row["audio"]
        filename = audio_data["path"]
        output_path = output_dir / filename

        if not output_path.exists():
            with open(output_path, "wb") as f:
                f.write(audio_data["bytes"])

        extracted_paths.append(str(output_path))

    return extracted_paths
def load_iemocap_metadata(parquet_files, audio_dir):
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

    for _, row in df.iterrows():
        emotion = emotion_map.get(row["major_emotion"])

        # Drop surprise and other.
        if emotion is None:
            continue

        filename = Path(row["file"]).name

        # IEMOCAP speaker = session + gender.
        # Example: Ses01F_impro01_F000.wav -> iemocap_Ses01F
        session_gender = filename[:6]

        rows.append(
            {
                "utterance_id": f"iemocap_{filename}",
                "corpus": "IEMOCAP",
                "speaker_id": f"iemocap_{session_gender}",
                "emotion": emotion,
                "audio_path": str(Path(audio_dir) / filename),
            }
        )
    return pd.DataFrame(rows, columns=METADATA_COLUMNS)
def load_cremad_metadata(root_dir):
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

    for audio_path in root.rglob("*.wav"):
        filename = audio_path.stem
        parts = filename.split("_")

        # CREMA-D filenames must contain 4 fields.
        if len(parts) != 4:
            continue

        actor_id = parts[0]
        emotion_code = parts[2]

        if emotion_code not in emotion_map:
            continue

        rows.append(
            {
                "utterance_id": f"cremad_{filename}",
                "corpus": "CREMA-D",
                "speaker_id": f"cremad_{actor_id}",
                "emotion": emotion_map[emotion_code],
                "audio_path": str(audio_path),
            }
        )

    return pd.DataFrame(rows, columns=METADATA_COLUMNS)
