export interface PredictionResponse {
  predicted_emotion: string;
  confidence: number;
  probabilities: Record<string, number>;
  processing_time_seconds: number;
}

export interface HealthResponse {
  status: string;
  model_loaded: boolean;
  checkpoint: string;
  emotion_labels: string[];
  error: string | null;
}

export interface ResearchMetricsResponse {
  models: Array<{
    name: string;
    metrics: {
      accuracy: number;
      macro_f1: number;
      weighted_f1: number;
    };
  }>;
  cross_corpus: Record<
    string,
    Record<
      string,
      {
        accuracy: number;
        macro_f1: number;
        weighted_f1: number;
      }
    >
  >;
}

const API_BASE = process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000";

export async function predictEmotion(file: File): Promise<PredictionResponse> {
  const form = new FormData();
  form.append("file", file);
  const res = await fetch(`${API_BASE}/predict`, { method: "POST", body: form });
  if (!res.ok) {
    const err = await res.json().catch(() => ({ detail: res.statusText }));
    throw new Error(err.detail || `HTTP ${res.status}`);
  }
  return res.json();
}

export async function getHealth(): Promise<HealthResponse> {
  const res = await fetch(`${API_BASE}/health`);
  if (!res.ok) throw new Error(`HTTP ${res.status}`);
  return res.json();
}

export async function getResearchMetrics(): Promise<ResearchMetricsResponse> {
  const res = await fetch(`${API_BASE}/research-metrics`);
  if (!res.ok) throw new Error(`HTTP ${res.status}`);
  return res.json();
}
