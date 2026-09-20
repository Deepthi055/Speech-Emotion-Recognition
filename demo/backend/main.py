import sys
import time
import json
import tempfile
import traceback
from math import gcd
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# Resolve project root (/mnt/shared/projects/Speech-Emotion-Recognition)
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.models.embedding_supcon_model import WavLMEmbeddingSupConModel
from src.datasets.metadata import EMOTION_LABELS

try:
    import soundfile as sf
    from scipy.signal import resample_poly
    from transformers import WavLMModel
except ImportError as e:
    raise RuntimeError(f"Missing dependency: {e}. Run: pip install -r requirements.txt")

MODEL_ID = "microsoft/wavlm-base"
TARGET_SR = 16000
CHECKPOINT_PATH = PROJECT_ROOT / "runs" / "wavlm_proposed_supcon" / "best_model.pt"

app = FastAPI(title="Speech Emotion Recognition API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["*"],
)

_wavlm: WavLMModel | None = None
_classifier: WavLMEmbeddingSupConModel | None = None
_device = torch.device("cpu")
_model_loaded = False
_load_error: str | None = None


def _load_models():
    global _wavlm, _classifier, _model_loaded, _load_error
    try:
        print(f"Loading WavLM backbone ({MODEL_ID}) on CPU...")
        wavlm = WavLMModel.from_pretrained(MODEL_ID)
        wavlm.eval()
        for p in wavlm.parameters():
            p.requires_grad = False
        _wavlm = wavlm

        print(f"Loading trained classifier from {CHECKPOINT_PATH}...")
        if not CHECKPOINT_PATH.exists():
            raise FileNotFoundError(f"Checkpoint not found: {CHECKPOINT_PATH}")

        classifier = WavLMEmbeddingSupConModel(
            in_dim=768,
            proj_dim=128,
            num_classes=len(EMOTION_LABELS),
            use_projection=True,
            use_classifier=True,
        )
        state = torch.load(CHECKPOINT_PATH, map_location=_device)
        load_res = classifier.load_state_dict(state, strict=True)
        classifier.eval()
        _classifier = classifier

        param_count = sum(p.numel() for p in classifier.parameters())
        print(f"Classifier state_dict matched strictly: {load_res}")
        print(f"Classifier parameter count: {param_count:,}")

        print("Exact emotion label-to-index mapping:")
        for idx, label in enumerate(EMOTION_LABELS):
            print(f"  {idx} -> {label}")

        _model_loaded = True
        print(f"Models loaded successfully. Emotion labels: {EMOTION_LABELS}")
    except Exception as e:
        _load_error = f"{type(e).__name__}: {e}"
        print(f"ERROR loading models: {_load_error}")


@app.on_event("startup")
def startup_event():
    _load_models()


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    checkpoint: str
    emotion_labels: list[str]
    error: str | None = None


class PredictionResponse(BaseModel):
    predicted_emotion: str
    confidence: float
    probabilities: dict[str, float]
    processing_time_seconds: float


@app.get("/health", response_model=HealthResponse)
def health():
    return HealthResponse(
        status="ready" if _model_loaded else "error",
        model_loaded=_model_loaded,
        checkpoint=str(CHECKPOINT_PATH),
        emotion_labels=EMOTION_LABELS,
        error=_load_error,
    )


@app.get("/research-metrics")
def get_research_metrics():
    in_domain_models = []
    final_dir = PROJECT_ROOT / "results" / "final"
    b_file = final_dir / "baseline_comparison.csv"
    if b_file.exists():
        with open(b_file, "r") as f:
            reader = csv.DictReader(f)
            for row in reader:
                in_domain_models.append({
                    "name": row.get("model_name", "Unknown"),
                    "metrics": {
                        "accuracy": float(row.get("war", 0.0)),
                        "uar": float(row.get("uar", 0.0)),
                        "macro_f1": float(row.get("macro_f1", 0.0)),
                        "weighted_f1": float(row.get("weighted_f1", 0.0)),
                    }
                })

    cross_corpus_dict: dict[str, dict[str, dict[str, float]]] = {}
    cc_file = final_dir / "cross_corpus_metrics.json"
    if cc_file.exists():
        with open(cc_file, "r") as f:
            raw_cc = json.load(f)
            for item in raw_cc:
                if not isinstance(item, dict):
                    continue
                setup = item.get("setup_key", "unknown")
                variant = item.get("variant_title", item.get("variant", "unknown"))
                if setup not in cross_corpus_dict:
                    cross_corpus_dict[setup] = {}
                cross_corpus_dict[setup][variant] = {
                    "accuracy": float(item.get("test_accuracy", 0.0)),
                    "uar": float(item.get("test_uar", 0.0)),
                    "macro_f1": float(item.get("test_f1_macro", 0.0)),
                    "weighted_f1": float(item.get("test_f1_weighted", 0.0)),
                }

    return {
        "models": in_domain_models,
        "cross_corpus": cross_corpus_dict,
    }


def _load_audio(file_bytes: bytes, filename: str) -> torch.Tensor:
    suffix = Path(filename).suffix.lower() if filename else ".wav"
    allowed = {".wav", ".mp3", ".flac", ".ogg", ".m4a", ".webm"}
    if suffix not in allowed:
        raise HTTPException(
            status_code=422,
            detail=f"Unsupported file type '{suffix}'. Supported: {', '.join(sorted(allowed))}",
        )
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
        tmp.write(file_bytes)
        tmp_path = Path(tmp.name)

    wav_tmp_path = tmp_path.with_suffix(".converted.wav")
    try:
        try:
            waveform, sr = sf.read(str(tmp_path), dtype="float32", always_2d=True)
        except Exception:
            # Fallback to ffmpeg for webm / compressed formats
            import subprocess
            cmd = ["ffmpeg", "-y", "-i", str(tmp_path), "-ar", "16000", "-ac", "1", str(wav_tmp_path)]
            res = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            if res.returncode != 0:
                raise HTTPException(status_code=422, detail="Could not decode audio with soundfile or ffmpeg.")
            waveform, sr = sf.read(str(wav_tmp_path), dtype="float32", always_2d=True)
    finally:
        tmp_path.unlink(missing_ok=True)
        wav_tmp_path.unlink(missing_ok=True)

    waveform = waveform.mean(axis=1)
    if waveform.size == 0:
        raise HTTPException(status_code=422, detail="Audio file appears to be empty.")
    if sr != TARGET_SR:
        g = gcd(sr, TARGET_SR)
        waveform = resample_poly(waveform, TARGET_SR // g, sr // g).astype(np.float32)
    return torch.from_numpy(waveform)


@torch.no_grad()
def _run_inference(waveform: torch.Tensor) -> np.ndarray:
    input_values = waveform.unsqueeze(0).to(_device)
    outputs = _wavlm(input_values)
    hidden = outputs.last_hidden_state
    embedding = hidden.mean(dim=1)
    result = _classifier(embedding)
    logits = result["logits"]
    probs = F.softmax(logits, dim=-1).squeeze(0).cpu().numpy()
    return probs


@app.post("/predict", response_model=PredictionResponse)
async def predict(file: UploadFile = File(...)):
    if not _model_loaded:
        raise HTTPException(status_code=503, detail=f"Model not loaded. {_load_error or ''}")

    t0 = time.perf_counter()

    raw = await file.read()
    if len(raw) == 0:
        raise HTTPException(status_code=422, detail="Uploaded file is empty.")

    waveform = _load_audio(raw, file.filename or "audio.wav")
    probs = _run_inference(waveform)

    pred_idx = int(np.argmax(probs))
    predicted_emotion = EMOTION_LABELS[pred_idx]
    confidence = float(probs[pred_idx])
    prob_dict = {label: float(probs[i]) for i, label in enumerate(EMOTION_LABELS)}
    elapsed = time.perf_counter() - t0

    return PredictionResponse(
        predicted_emotion=predicted_emotion,
        confidence=confidence,
        probabilities=prob_dict,
        processing_time_seconds=round(elapsed, 3),
    )
