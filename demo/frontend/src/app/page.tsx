"use client";

import { useState, useEffect, useRef } from "react";
import { useAudioRecorder } from "@/hooks/useAudioRecorder";
import { FileUploader } from "@/components/FileUploader";
import { predictEmotion, getHealth, getResearchMetrics, PredictionResponse, ResearchMetricsResponse } from "@/lib/api";

const EMOTION_COLORS: Record<string, string> = {
  angry: "#ef4444",
  disgust: "#10b981",
  fear: "#a855f7",
  happy: "#f59e0b",
  neutral: "#6b7280",
  sad: "#3b82f6",
};

export default function Home() {
  const [activeTab, setActiveTab] = useState<"demo" | "research">("demo");
  const [selectedFile, setSelectedFile] = useState<File | null>(null);
  const [audioUrl, setAudioUrl] = useState<string | null>(null);
  const [loading, setLoading] = useState(false);
  const [prediction, setPrediction] = useState<PredictionResponse | null>(null);
  const [error, setError] = useState<string | null>(null);

  const [health, setHealth] = useState<{ status: string; model_loaded: boolean } | null>(null);
  const [researchData, setResearchData] = useState<ResearchMetricsResponse | null>(null);

  const recorder = useAudioRecorder();

  useEffect(() => {
    getHealth().then(setHealth).catch(() => setHealth({ status: "offline", model_loaded: false }));
    getResearchMetrics().then(setResearchData).catch(() => {});
  }, []);

  const handleAudioBlob = (blob: Blob) => {
    const file = new File([blob], "recorded_audio.wav", { type: blob.type || "audio/wav" });
    const url = URL.createObjectURL(blob);
    setSelectedFile(file);
    setAudioUrl(url);
    setPrediction(null);
    setError(null);
  };

  const handleUploadFile = (file: File, url: string) => {
    setSelectedFile(file);
    setAudioUrl(url);
    setPrediction(null);
    setError(null);
  };

  const handlePredict = async () => {
    let fileToUpload = selectedFile;
    if (!fileToUpload && recorder.audioBlob) {
      fileToUpload = new File([recorder.audioBlob], "recorded_audio.wav", { type: recorder.audioBlob.type || "audio/wav" });
    }
    if (!fileToUpload) return;

    setLoading(true);
    setError(null);
    try {
      const res = await predictEmotion(fileToUpload);
      setPrediction(res);
    } catch (err: any) {
      setError(err.message || "Failed to analyze audio");
    } finally {
      setLoading(false);
    }
  };

  return (
    <div style={{ maxWidth: "1000px", margin: "0 auto", padding: "2rem 1.5rem" }}>
      {/* Header */}
      <header style={{ marginBottom: "2.5rem", borderBottom: "1px solid var(--bg-border)", paddingBottom: "1.5rem" }}>
        <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center" }}>
          <div>
            <h1 style={{ fontSize: "1.75rem", fontWeight: 700, margin: 0, letterSpacing: "-0.02em" }}>
              Speech Emotion Recognition
            </h1>
            <p style={{ color: "var(--text-secondary)", marginTop: "0.4rem", fontSize: "0.95rem" }}>
              Frozen WavLM + Speaker-Corpus Aware Supervised Contrastive Learning
            </p>
          </div>
          <div style={{ display: "flex", alignItems: "center", gap: "0.5rem", fontSize: "0.85rem" }}>
            <span
              style={{
                display: "inline-block",
                width: "8px",
                height: "8px",
                borderRadius: "50%",
                background: health?.model_loaded ? "#10b981" : "#ef4444",
              }}
            />
            <span style={{ color: "var(--text-secondary)" }}>
              {health?.model_loaded ? "Model Ready" : health?.status === "offline" ? "Backend Offline" : "Loading Model..."}
            </span>
          </div>
        </div>

        {/* Tab Switcher */}
        <div style={{ display: "flex", gap: "1rem", marginTop: "1.5rem" }}>
          <button
            onClick={() => setActiveTab("demo")}
            style={{
              padding: "0.5rem 1rem",
              borderRadius: "6px",
              border: "1px solid",
              borderColor: activeTab === "demo" ? "var(--accent)" : "var(--bg-border)",
              background: activeTab === "demo" ? "var(--accent-dim)" : "transparent",
              color: activeTab === "demo" ? "var(--accent)" : "var(--text-secondary)",
              cursor: "pointer",
              fontWeight: 500,
            }}
          >
            Live Demo / Predict
          </button>
          <button
            onClick={() => setActiveTab("research")}
            style={{
              padding: "0.5rem 1rem",
              borderRadius: "6px",
              border: "1px solid",
              borderColor: activeTab === "research" ? "var(--accent)" : "var(--bg-border)",
              background: activeTab === "research" ? "var(--accent-dim)" : "transparent",
              color: activeTab === "research" ? "var(--accent)" : "var(--text-secondary)",
              cursor: "pointer",
              fontWeight: 500,
            }}
          >
            Research & Benchmarks
          </button>
        </div>
      </header>

      {/* Main Content */}
      {activeTab === "demo" && (
        <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: "2rem" }}>
          {/* Input Panel */}
          <div style={{ display: "flex", flexDirection: "column", gap: "1.5rem" }}>
            {/* Record Section */}
            <div
              style={{
                background: "var(--bg-surface)",
                border: "1px solid var(--bg-border)",
                borderRadius: "10px",
                padding: "1.5rem",
              }}
            >
              <h3 style={{ margin: "0 0 1rem 0", fontSize: "1.1rem", fontWeight: 600 }}>Record Audio</h3>
              <div style={{ display: "flex", alignItems: "center", gap: "1rem" }}>
                {recorder.state !== "recording" ? (
                  <button
                    onClick={recorder.startRecording}
                    style={{
                      padding: "0.6rem 1.2rem",
                      borderRadius: "6px",
                      background: "#ef4444",
                      color: "#fff",
                      border: "none",
                      fontWeight: 600,
                      cursor: "pointer",
                    }}
                  >
                    🎤 Start Recording
                  </button>
                ) : (
                  <button
                    onClick={() => {
                      recorder.stopRecording().then(handleAudioBlob);
                    }}
                    style={{
                      padding: "0.6rem 1.2rem",
                      borderRadius: "6px",
                      background: "#3b82f6",
                      color: "#fff",
                      border: "none",
                      fontWeight: 600,
                      cursor: "pointer",
                    }}
                  >
                    ⏹ Stop Recording ({recorder.duration}s)
                  </button>
                )}
                {recorder.state === "recording" && (
                  <span style={{ color: "#ef4444", fontSize: "0.85rem", animation: "pulse 1s infinite" }}>● Live</span>
                )}
              </div>
            </div>

            {/* Upload Section */}
            <div
              style={{
                background: "var(--bg-surface)",
                border: "1px solid var(--bg-border)",
                borderRadius: "10px",
                padding: "1.5rem",
              }}
            >
              <h3 style={{ margin: "0 0 1rem 0", fontSize: "1.1rem", fontWeight: 600 }}>Upload Audio File</h3>
              <FileUploader onFile={handleUploadFile} />
            </div>

            {/* Audio Preview & Predict Button */}
            {(audioUrl || selectedFile) && (
              <div
                style={{
                  background: "var(--bg-surface)",
                  border: "1px solid var(--bg-border)",
                  borderRadius: "10px",
                  padding: "1.5rem",
                }}
              >
                <h4 style={{ margin: "0 0 0.8rem 0", color: "var(--text-secondary)", fontSize: "0.9rem" }}>
                  Selected Audio: {selectedFile?.name || "Recorded Audio"}
                </h4>
                {audioUrl && <audio controls src={audioUrl} style={{ width: "100%", marginBottom: "1rem" }} />}
                <button
                  onClick={handlePredict}
                  disabled={loading || !health?.model_loaded}
                  style={{
                    width: "100%",
                    padding: "0.75rem",
                    borderRadius: "6px",
                    background: loading ? "var(--bg-border)" : "var(--accent)",
                    color: "#fff",
                    border: "none",
                    fontWeight: 600,
                    fontSize: "1rem",
                    cursor: loading ? "not-allowed" : "pointer",
                  }}
                >
                  {loading ? "Extracting WavLM Features & Classifying..." : "Analyze Emotion"}
                </button>
              </div>
            )}

            {error && (
              <div
                style={{
                  padding: "1rem",
                  borderRadius: "6px",
                  background: "rgba(239, 68, 68, 0.1)",
                  border: "1px solid rgba(239, 68, 68, 0.3)",
                  color: "#ef4444",
                  fontSize: "0.9rem",
                }}
              >
                {error}
              </div>
            )}
          </div>

          {/* Results Panel */}
          <div
            style={{
              background: "var(--bg-surface)",
              border: "1px solid var(--bg-border)",
              borderRadius: "10px",
              padding: "1.5rem",
              display: "flex",
              flexDirection: "column",
            }}
          >
            <h3 style={{ margin: "0 0 1.5rem 0", fontSize: "1.1rem", fontWeight: 600 }}>Prediction Result</h3>

            {!prediction && !loading && (
              <div
                style={{
                  flex: 1,
                  display: "flex",
                  alignItems: "center",
                  justifyContent: "center",
                  color: "var(--text-muted)",
                  textAlign: "center",
                  fontSize: "0.95rem",
                }}
              >
                Record or upload an audio file and click "Analyze Emotion" to view model predictions.
              </div>
            )}

            {loading && (
              <div style={{ flex: 1, display: "flex", flexDirection: "column", alignItems: "center", justifyContent: "center", gap: "1rem" }}>
                <div
                  style={{
                    width: "32px",
                    height: "32px",
                    border: "3px solid var(--bg-border)",
                    borderTopColor: "var(--accent)",
                    borderRadius: "50%",
                    animation: "spin 1s linear infinite",
                  }}
                />
                <span style={{ color: "var(--text-secondary)", fontSize: "0.9rem" }}>Passing audio through frozen WavLM encoder...</span>
              </div>
            )}

            {prediction && (
              <div style={{ display: "flex", flexDirection: "column", gap: "1.5rem" }}>
                {/* Predicted Emotion Badge */}
                <div
                  style={{
                    padding: "1.5rem",
                    borderRadius: "8px",
                    background: "rgba(255, 255, 255, 0.03)",
                    border: `1px solid ${EMOTION_COLORS[prediction.predicted_emotion] || "var(--accent)"}`,
                    textAlign: "center",
                  }}
                >
                  <span style={{ fontSize: "0.85rem", color: "var(--text-secondary)", textTransform: "uppercase", letterSpacing: "0.05em" }}>
                    Detected Emotion
                  </span>
                  <div
                    style={{
                      fontSize: "2.2rem",
                      fontWeight: 700,
                      color: EMOTION_COLORS[prediction.predicted_emotion] || "#fff",
                      textTransform: "capitalize",
                      margin: "0.2rem 0",
                    }}
                  >
                    {prediction.predicted_emotion}
                  </div>
                  <div style={{ fontSize: "0.9rem", color: "var(--text-secondary)" }}>
                    Confidence: {(prediction.confidence * 100).toFixed(1)}% | Latency: {prediction.processing_time_seconds.toFixed(2)}s
                  </div>
                </div>

                {/* Probability Distribution */}
                <div>
                  <h4 style={{ margin: "0 0 1rem 0", fontSize: "0.95rem", color: "var(--text-secondary)" }}>Emotion Probabilities</h4>
                  <div style={{ display: "flex", flexDirection: "column", gap: "0.75rem" }}>
                    {Object.entries(prediction.probabilities)
                      .sort(([, a], [, b]) => b - a)
                      .map(([emo, prob]) => (
                        <div key={emo} style={{ display: "flex", flexDirection: "column", gap: "0.25rem" }}>
                          <div style={{ display: "flex", justifyContent: "space-between", fontSize: "0.85rem" }}>
                            <span style={{ textTransform: "capitalize", fontWeight: emo === prediction.predicted_emotion ? 600 : 400 }}>{emo}</span>
                            <span style={{ color: "var(--text-secondary)" }}>{(prob * 100).toFixed(1)}%</span>
                          </div>
                          <div
                            style={{
                              width: "100%",
                              height: "6px",
                              borderRadius: "3px",
                              background: "var(--bg-border)",
                              overflow: "hidden",
                            }}
                          >
                            <div
                              style={{
                                width: `${prob * 100}%`,
                                height: "100%",
                                background: EMOTION_COLORS[emo] || "var(--accent)",
                                transition: "width 0.3s ease",
                              }}
                            />
                          </div>
                        </div>
                      ))}
                  </div>
                </div>
              </div>
            )}
          </div>
        </div>
      )}

      {/* Research & Benchmarks Tab */}
      {activeTab === "research" && (
        <div style={{ display: "flex", flexDirection: "column", gap: "2rem" }}>
          <div style={{ background: "var(--bg-surface)", border: "1px solid var(--bg-border)", borderRadius: "10px", padding: "1.5rem" }}>
            <h3 style={{ margin: "0 0 0.5rem 0", fontSize: "1.2rem", fontWeight: 600 }}>Proposed Architecture</h3>
            <p style={{ color: "var(--text-secondary)", fontSize: "0.95rem", lineHeight: 1.6 }}>
              Our method uses a frozen <strong>WavLM-base</strong> audio backbone to extract frame-level speech features, which are mean-pooled into 768-dim embeddings. We train a linear classifier using <strong>Speaker + Corpus-aware Supervised Contrastive Learning (SupCon)</strong> to learn representations robust against speaker variation and cross-corpus domain shifts.
            </p>
          </div>

          {researchData ? (
            <>
              {/* In-domain Model Comparison */}
              <div style={{ background: "var(--bg-surface)", border: "1px solid var(--bg-border)", borderRadius: "10px", padding: "1.5rem" }}>
                <h3 style={{ margin: "0 0 1rem 0", fontSize: "1.1rem", fontWeight: 600 }}>In-Domain Benchmarks (CREMA-D, RAVDESS, IEMOCAP)</h3>
                <table style={{ width: "100%", borderCollapse: "collapse", fontSize: "0.9rem", textAlign: "left" }}>
                  <thead>
                    <tr style={{ borderBottom: "1px solid var(--bg-border)", color: "var(--text-secondary)" }}>
                      <th style={{ padding: "0.75rem" }}>Model Architecture</th>
                      <th style={{ padding: "0.75rem" }}>Accuracy</th>
                      <th style={{ padding: "0.75rem" }}>Macro F1</th>
                      <th style={{ padding: "0.75rem" }}>Weighted F1</th>
                    </tr>
                  </thead>
                  <tbody>
                    {researchData.models.map((m, idx) => (
                      <tr
                        key={m.name}
                        style={{
                          borderBottom: "1px solid var(--bg-border)",
                          background: m.name.includes("Proposed") ? "var(--accent-dim)" : "transparent",
                          fontWeight: m.name.includes("Proposed") ? 600 : 400,
                        }}
                      >
                        <td style={{ padding: "0.75rem" }}>{m.name}</td>
                        <td style={{ padding: "0.75rem" }}>{(m.metrics.accuracy * 100).toFixed(2)}%</td>
                        <td style={{ padding: "0.75rem" }}>{(m.metrics.macro_f1 * 100).toFixed(2)}%</td>
                        <td style={{ padding: "0.75rem" }}>{(m.metrics.weighted_f1 * 100).toFixed(2)}%</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>

              {/* Cross-Corpus Results */}
              <div style={{ background: "var(--bg-surface)", border: "1px solid var(--bg-border)", borderRadius: "10px", padding: "1.5rem" }}>
                <h3 style={{ margin: "0 0 1rem 0", fontSize: "1.1rem", fontWeight: 600 }}>Cross-Corpus Generalization (Zero-Shot Unseen Corpus)</h3>
                <table style={{ width: "100%", borderCollapse: "collapse", fontSize: "0.9rem", textAlign: "left" }}>
                  <thead>
                    <tr style={{ borderBottom: "1px solid var(--bg-border)", color: "var(--text-secondary)" }}>
                      <th style={{ padding: "0.75rem" }}>Train Setup → Test Target</th>
                      <th style={{ padding: "0.75rem" }}>WavLM Baseline F1</th>
                      <th style={{ padding: "0.75rem" }}>Proposed SupCon F1</th>
                      <th style={{ padding: "0.75rem" }}>Improvement</th>
                    </tr>
                  </thead>
                  <tbody>
                    {Object.entries(researchData.cross_corpus).map(([split, data]) => {
                      const ceF1 = data["WavLM + CE"]?.macro_f1 || 0;
                      const propF1 = data["Speaker + Corpus-aware SupCon (Proposed)"]?.macro_f1 || 0;
                      const diff = (propF1 - ceF1) * 100;
                      return (
                        <tr key={split} style={{ borderBottom: "1px solid var(--bg-border)" }}>
                          <td style={{ padding: "0.75rem" }}>{split.replace("_to_", " → ").toUpperCase()}</td>
                          <td style={{ padding: "0.75rem" }}>{(ceF1 * 100).toFixed(2)}%</td>
                          <td style={{ padding: "0.75rem", fontWeight: 600, color: "var(--accent)" }}>{(propF1 * 100).toFixed(2)}%</td>
                          <td style={{ padding: "0.75rem", color: diff >= 0 ? "#10b981" : "#ef4444" }}>
                            {diff >= 0 ? `+${diff.toFixed(2)}%` : `${diff.toFixed(2)}%`}
                          </td>
                        </tr>
                      );
                    })}
                  </tbody>
                </table>
              </div>
            </>
          ) : (
            <div style={{ color: "var(--text-muted)" }}>Loading research benchmarks...</div>
          )}
        </div>
      )}
    </div>
  );
}
