"use client";

import { useState, useEffect } from "react";
import { FileUploader } from "@/components/FileUploader";
import { predictEmotion, getHealth, getResearchMetrics, PredictionResponse, ResearchMetricsResponse } from "@/lib/api";

export default function Home() {
  const [selectedFile, setSelectedFile] = useState<File | null>(null);
  const [audioUrl, setAudioUrl] = useState<string | null>(null);
  const [loading, setLoading] = useState(false);
  const [prediction, setPrediction] = useState<PredictionResponse | null>(null);
  const [error, setError] = useState<string | null>(null);

  const [health, setHealth] = useState<{ status: string; model_loaded: boolean } | null>(null);
  const [metricsData, setMetricsData] = useState<ResearchMetricsResponse | null>(null);

  useEffect(() => {
    getHealth().then(setHealth).catch(() => setHealth({ status: "offline", model_loaded: false }));
    getResearchMetrics().then(setMetricsData).catch(() => {});
  }, []);

  const handleUploadFile = (file: File, url: string) => {
    if (audioUrl) URL.revokeObjectURL(audioUrl);
    setSelectedFile(file);
    setAudioUrl(url);
    setPrediction(null);
    setError(null);
  };

  const handleClear = () => {
    if (audioUrl) URL.revokeObjectURL(audioUrl);
    setSelectedFile(null);
    setAudioUrl(null);
    setPrediction(null);
    setError(null);
  };

  const handlePredict = async () => {
    if (!selectedFile) return;

    setLoading(true);
    setError(null);
    try {
      const res = await predictEmotion(selectedFile);
      setPrediction(res);
    } catch (err: any) {
      setError(err.message || "Execution error during inference pass.");
    } finally {
      setLoading(false);
    }
  };

  return (
    <div style={{ minHeight: "100vh", display: "flex", flexDirection: "column", background: "var(--bg)" }}>
      {/* Top Header Bar */}
      <header
        style={{
          borderBottom: "1px solid var(--border)",
          background: "var(--bg-subtle)",
          padding: "0.75rem 1.5rem",
        }}
      >
        <div
          style={{
            width: "100%",
            display: "flex",
            justifyContent: "space-between",
            alignItems: "center",
          }}
        >
          <div style={{ display: "flex", alignItems: "center", gap: "1rem" }}>
            <span
              className="font-mono"
              style={{
                fontSize: "0.75rem",
                fontWeight: 600,
                letterSpacing: "0.08em",
                color: "var(--text-primary)",
                background: "var(--bg-element)",
                padding: "2px 8px",
                border: "1px solid var(--border)",
                borderRadius: "2px",
              }}
            >
              SER / AUD-01
            </span>
            <div>
              <h1
                style={{
                  fontSize: "0.95rem",
                  fontWeight: 600,
                  letterSpacing: "-0.01em",
                  color: "var(--text-primary)",
                  display: "flex",
                  alignItems: "center",
                  gap: "0.5rem",
                }}
              >
                Speech Emotion Intelligence System
              </h1>
            </div>
          </div>

          <div
            className="font-mono"
            style={{
              fontSize: "0.75rem",
              color: "var(--text-muted)",
              display: "flex",
              alignItems: "center",
              gap: "1.25rem",
            }}
          >
            <span>BACKBONE: <strong style={{ color: "var(--text-secondary)", fontWeight: 500 }}>WavLM-Base</strong></span>
            <span>EMBED_DIM: <strong style={{ color: "var(--text-secondary)", fontWeight: 500 }}>768</strong></span>
            <span>
              STATUS:{" "}
              <strong style={{ color: health?.model_loaded ? "#34d399" : "#f43f5e", fontWeight: 600 }}>
                {health?.model_loaded ? "ONLINE" : "OFFLINE"}
              </strong>
            </span>
          </div>
        </div>
      </header>

      {/* Main Workspace (Full Width, Compact Margins) */}
      <main style={{ flex: 1, width: "100%", padding: "1.25rem 1.5rem" }}>
        
        {/* Dual Workspace Panels */}
        <div
          style={{
            display: "grid",
            gridTemplateColumns: "1fr 1fr",
            gap: "1px",
            background: "var(--border)",
            border: "1px solid var(--border)",
            borderRadius: "4px",
            overflow: "hidden",
          }}
        >
          {/* Left Panel: Input & Controls */}
          <div
            style={{
              background: "var(--bg-panel)",
              padding: "1.25rem",
              display: "flex",
              flexDirection: "column",
              justifyContent: "space-between",
              minHeight: "420px",
            }}
          >
            <div style={{ display: "flex", flexDirection: "column", gap: "1rem" }}>
              <div
                style={{
                  display: "flex",
                  justifyContent: "space-between",
                  alignItems: "center",
                  paddingBottom: "0.5rem",
                  borderBottom: "1px solid var(--border-subtle)",
                }}
              >
                <span
                  className="font-mono"
                  style={{
                    fontSize: "0.7rem",
                    fontWeight: 600,
                    letterSpacing: "0.08em",
                    color: "var(--text-muted)",
                    textTransform: "uppercase",
                  }}
                >
                  AUDIO INPUT PAYLOAD
                </span>
                {selectedFile && (
                  <button
                    onClick={handleClear}
                    className="font-mono"
                    style={{
                      background: "transparent",
                      border: "none",
                      color: "var(--text-muted)",
                      fontSize: "0.7rem",
                      cursor: "pointer",
                      textTransform: "uppercase",
                      letterSpacing: "0.05em",
                    }}
                    onMouseOver={(e) => (e.currentTarget.style.color = "var(--text-primary)")}
                    onMouseOut={(e) => (e.currentTarget.style.color = "var(--text-muted)")}
                  >
                    [ CLEAR FILE ]
                  </button>
                )}
              </div>

              {/* Upload Dropzone (Always visible to allow easy re-uploading) */}
              <FileUploader onFile={handleUploadFile} />

              {/* Active Audio Metadata & Waveform Player */}
              {selectedFile && (
                <div
                  className="animate-fade"
                  style={{
                    background: "var(--bg-subtle)",
                    border: "1px solid var(--border)",
                    borderRadius: "3px",
                    padding: "0.85rem 1rem",
                    display: "flex",
                    flexDirection: "column",
                    gap: "0.75rem",
                  }}
                >
                  <div style={{ display: "flex", justifyContent: "space-between", alignItems: "baseline" }}>
                    <span className="font-mono" style={{ fontSize: "0.8rem", color: "var(--text-primary)", fontWeight: 500, overflow: "hidden", textOverflow: "ellipsis", whiteSpace: "nowrap", maxWidth: "340px" }}>
                      {selectedFile.name}
                    </span>
                    <span className="font-mono" style={{ fontSize: "0.7rem", color: "var(--text-muted)" }}>
                      {(selectedFile.size / 1024).toFixed(1)} KB
                    </span>
                  </div>

                  <div className="font-mono" style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: "0.5rem", fontSize: "0.7rem", color: "var(--text-muted)", paddingTop: "0.35rem", borderTop: "1px solid var(--border-subtle)" }}>
                    <span>TARGET_SR: 16000 Hz</span>
                    <span>CHANNELS: 1 (MONO)</span>
                  </div>

                  {audioUrl && (
                    <audio
                      controls
                      src={audioUrl}
                      style={{
                        width: "100%",
                        height: "36px",
                        borderRadius: "2px",
                      }}
                    />
                  )}
                </div>
              )}
            </div>

            <div style={{ marginTop: "1.25rem" }}>
              {error && (
                <div
                  className="font-mono"
                  style={{
                    padding: "0.6rem 0.8rem",
                    borderRadius: "2px",
                    background: "rgba(244, 63, 94, 0.08)",
                    border: "1px solid rgba(244, 63, 94, 0.25)",
                    color: "var(--error)",
                    fontSize: "0.75rem",
                    marginBottom: "0.75rem",
                  }}
                >
                  ERR: {error}
                </div>
              )}

              <button
                onClick={handlePredict}
                disabled={loading || !selectedFile || !health?.model_loaded}
                className="font-mono"
                style={{
                  width: "100%",
                  padding: "0.75rem",
                  borderRadius: "2px",
                  background: loading ? "var(--bg-element)" : !selectedFile ? "var(--bg-subtle)" : "var(--text-primary)",
                  color: loading || !selectedFile ? "var(--text-muted)" : "var(--bg)",
                  border: "1px solid var(--border)",
                  fontWeight: 600,
                  fontSize: "0.75rem",
                  letterSpacing: "0.05em",
                  textTransform: "uppercase",
                  cursor: loading || !selectedFile || !health?.model_loaded ? "not-allowed" : "pointer",
                  display: "flex",
                  alignItems: "center",
                  justifyContent: "center",
                  gap: "0.5rem",
                  transition: "all 0.12s ease",
                }}
              >
                {loading && (
                  <span
                    style={{
                      width: "12px",
                      height: "12px",
                      border: "2px solid var(--text-muted)",
                      borderTopColor: "var(--text-primary)",
                      borderRadius: "50%",
                      animation: "spin 0.6s linear infinite",
                    }}
                  />
                )}
                {loading ? "COMPUTING EMBEDDINGS & INFERENCE..." : "EXECUTE MODEL INFERENCE"}
              </button>
            </div>
          </div>

          {/* Right Panel: Output & Probabilities */}
          <div
            style={{
              background: "var(--bg-panel)",
              padding: "1.25rem",
              display: "flex",
              flexDirection: "column",
              justifyContent: "space-between",
              minHeight: "420px",
            }}
          >
            <div>
              <div
                style={{
                  display: "flex",
                  justifyContent: "space-between",
                  alignItems: "center",
                  marginBottom: "1rem",
                  paddingBottom: "0.5rem",
                  borderBottom: "1px solid var(--border-subtle)",
                }}
              >
                <span
                  className="font-mono"
                  style={{
                    fontSize: "0.7rem",
                    fontWeight: 600,
                    letterSpacing: "0.08em",
                    color: "var(--text-muted)",
                    textTransform: "uppercase",
                  }}
                >
                  CLASSIFICATION OUTPUT
                </span>
                {prediction && (
                  <span className="font-mono" style={{ fontSize: "0.7rem", color: "var(--text-muted)" }}>
                    LATENCY: {prediction.processing_time_seconds.toFixed(3)}s
                  </span>
                )}
              </div>

              {!prediction && !loading && (
                <div
                  style={{
                    height: "300px",
                    display: "flex",
                    alignItems: "center",
                    justifyContent: "center",
                    color: "var(--text-muted)",
                    fontSize: "0.8rem",
                  }}
                  className="font-mono"
                >
                  [ AWAITING INPUT INFERENCE PASS ]
                </div>
              )}

              {loading && (
                <div style={{ height: "300px", display: "flex", flexDirection: "column", alignItems: "center", justifyContent: "center", gap: "0.75rem" }}>
                  <div
                    style={{
                      width: "24px",
                      height: "24px",
                      border: "2px solid var(--border)",
                      borderTopColor: "var(--accent)",
                      borderRadius: "50%",
                      animation: "spin 0.6s linear infinite",
                    }}
                  />
                  <span className="font-mono" style={{ color: "var(--text-muted)", fontSize: "0.75rem" }}>
                    EXTRACTING WAVEFORM FEATURES (WAVLM)...
                  </span>
                </div>
              )}

              {prediction && (
                <div className="animate-fade" style={{ display: "flex", flexDirection: "column", gap: "1.25rem" }}>
                  {/* Primary Emotion Result Banner */}
                  <div
                    style={{
                      background: "var(--bg-subtle)",
                      border: "1px solid var(--border)",
                      borderRadius: "3px",
                      padding: "1rem 1.25rem",
                      display: "flex",
                      justifyContent: "space-between",
                      alignItems: "center",
                    }}
                  >
                    <div>
                      <div className="font-mono" style={{ fontSize: "0.65rem", color: "var(--text-muted)", textTransform: "uppercase", letterSpacing: "0.08em" }}>
                        PREDICTED CLASS
                      </div>
                      <div
                        style={{
                          fontSize: "1.75rem",
                          fontWeight: 600,
                          color: "var(--text-primary)",
                          textTransform: "uppercase",
                          letterSpacing: "0.02em",
                          marginTop: "2px",
                        }}
                      >
                        {prediction.predicted_emotion}
                      </div>
                    </div>

                    <div className="font-mono" style={{ textAlign: "right" }}>
                      <div style={{ fontSize: "0.65rem", color: "var(--text-muted)", textTransform: "uppercase", letterSpacing: "0.08em" }}>
                        CONFIDENCE
                      </div>
                      <div style={{ fontSize: "1.5rem", fontWeight: 600, color: "var(--accent)", marginTop: "2px" }}>
                        {(prediction.confidence * 100).toFixed(2)}%
                      </div>
                    </div>
                  </div>

                  {/* Probability Distribution Table */}
                  <div>
                    <div className="font-mono" style={{ fontSize: "0.7rem", color: "var(--text-muted)", textTransform: "uppercase", letterSpacing: "0.08em", marginBottom: "0.6rem" }}>
                      CLASS PROBABILITY DISTRIBUTION
                    </div>
                    <div style={{ display: "flex", flexDirection: "column", gap: "0.45rem" }}>
                      {Object.entries(prediction.probabilities)
                        .sort(([, a], [, b]) => b - a)
                        .map(([emo, prob]) => {
                          const isTop = emo === prediction.predicted_emotion;
                          return (
                            <div key={emo} style={{ display: "grid", gridTemplateColumns: "80px 1fr 60px", alignItems: "center", gap: "0.75rem" }}>
                              <span
                                className="font-mono"
                                style={{
                                  fontSize: "0.75rem",
                                  textTransform: "uppercase",
                                  color: isTop ? "var(--text-primary)" : "var(--text-muted)",
                                  fontWeight: isTop ? 600 : 400,
                                }}
                              >
                                {emo}
                              </span>
                              <div
                                style={{
                                  height: "4px",
                                  background: "var(--bg-subtle)",
                                  border: "1px solid var(--border-subtle)",
                                  borderRadius: "1px",
                                  overflow: "hidden",
                                }}
                              >
                                <div
                                  style={{
                                    width: `${prob * 100}%`,
                                    height: "100%",
                                    background: isTop ? "var(--accent)" : "var(--text-muted)",
                                    transition: "width 0.25s ease",
                                  }}
                                />
                              </div>
                              <span
                                className="font-mono"
                                style={{
                                  fontSize: "0.75rem",
                                  textAlign: "right",
                                  color: isTop ? "var(--text-primary)" : "var(--text-muted)",
                                  fontWeight: isTop ? 600 : 400,
                                }}
                              >
                                {(prob * 100).toFixed(1)}%
                              </span>
                            </div>
                          );
                        })}
                    </div>
                  </div>
                </div>
              )}
            </div>
          </div>
        </div>

        {/* Integrated Performance Section */}
        {metricsData && (
          <section
            style={{
              marginTop: "1.25rem",
              background: "var(--bg-panel)",
              border: "1px solid var(--border)",
              borderRadius: "4px",
              overflow: "hidden",
            }}
          >
            <div
              style={{
                padding: "0.75rem 1.25rem",
                borderBottom: "1px solid var(--border)",
                background: "var(--bg-subtle)",
                display: "flex",
                justifyContent: "space-between",
                alignItems: "center",
              }}
            >
              <span
                className="font-mono"
                style={{
                  fontSize: "0.7rem",
                  fontWeight: 600,
                  letterSpacing: "0.08em",
                  color: "var(--text-muted)",
                  textTransform: "uppercase",
                }}
              >
                MODEL PERFORMANCE EVALUATION & GENERALIZATION MATRIX
              </span>
              <span className="font-mono" style={{ fontSize: "0.7rem", color: "var(--text-muted)" }}>
                TEST SAMPLES: 3,234 | CORPORA: CREMA-D, RAVDESS, IEMOCAP
              </span>
            </div>

            <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: "1px", background: "var(--border)" }}>
              {/* In-domain Benchmarks */}
              <div style={{ background: "var(--bg-panel)", padding: "1.25rem" }}>
                <div className="font-mono" style={{ fontSize: "0.75rem", fontWeight: 600, color: "var(--text-primary)", marginBottom: "0.75rem", letterSpacing: "0.04em" }}>
                  IN-DOMAIN BENCHMARKS
                </div>
                <table className="tech-table">
                  <thead>
                    <tr>
                      <th>MODEL VARIANT</th>
                      <th>ACCURACY</th>
                      <th>MACRO F1</th>
                      <th>WEIGHTED F1</th>
                    </tr>
                  </thead>
                  <tbody>
                    {metricsData.models.map((m) => {
                      const isMain = m.name.includes("Proposed") || m.name.includes("Corpus-Aware");
                      return (
                        <tr key={m.name} className={isMain ? "highlight" : ""}>
                          <td style={{ fontWeight: isMain ? 500 : 400 }}>{m.name}</td>
                          <td className="font-mono">{(m.metrics.accuracy * 100).toFixed(2)}%</td>
                          <td className="font-mono">{(m.metrics.macro_f1 * 100).toFixed(2)}%</td>
                          <td className="font-mono">{(m.metrics.weighted_f1 * 100).toFixed(2)}%</td>
                        </tr>
                      );
                    })}
                  </tbody>
                </table>
              </div>

              {/* Cross-Corpus Results */}
              <div style={{ background: "var(--bg-panel)", padding: "1.25rem" }}>
                <div className="font-mono" style={{ fontSize: "0.75rem", fontWeight: 600, color: "var(--text-primary)", marginBottom: "0.75rem", letterSpacing: "0.04em" }}>
                  CROSS-CORPUS DOMAIN SHIFT
                </div>
                <table className="tech-table">
                  <thead>
                    <tr>
                      <th>DOMAIN TRANSFER</th>
                      <th>BASELINE F1</th>
                      <th>MODEL F1</th>
                      <th>DELTA</th>
                    </tr>
                  </thead>
                  <tbody>
                    {Object.entries(metricsData.cross_corpus).map(([split, data]) => {
                      const ceF1 = data["WavLM + CE"]?.macro_f1 || 0;
                      const propF1 = data["Speaker + Corpus-aware SupCon (Proposed)"]?.macro_f1 || 0;
                      const diff = (propF1 - ceF1) * 100;
                      return (
                        <tr key={split}>
                          <td className="font-mono">{split.replace("_to_", " -> ").toUpperCase()}</td>
                          <td className="font-mono">{(ceF1 * 100).toFixed(2)}%</td>
                          <td className="font-mono" style={{ color: "var(--text-primary)", fontWeight: 500 }}>{(propF1 * 100).toFixed(2)}%</td>
                          <td className="font-mono" style={{ color: diff >= 0 ? "#34d399" : "#f43f5e" }}>
                            {diff >= 0 ? `+${diff.toFixed(2)}%` : `${diff.toFixed(2)}%`}
                          </td>
                        </tr>
                      );
                    })}
                  </tbody>
                </table>
              </div>
            </div>
          </section>
        )}
      </main>
    </div>
  );
}
