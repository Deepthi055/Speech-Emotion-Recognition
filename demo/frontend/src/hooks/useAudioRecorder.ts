"use client";

import { useRef, useState, useCallback, useEffect } from "react";

export type RecorderState = "idle" | "recording" | "done";

export function useAudioRecorder() {
  const [state, setState] = useState<RecorderState>("idle");
  const [audioUrl, setAudioUrl] = useState<string | null>(null);
  const [audioBlob, setAudioBlob] = useState<Blob | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [duration, setDuration] = useState(0);

  const mediaRef = useRef<MediaRecorder | null>(null);
  const chunksRef = useRef<Blob[]>([]);
  const timerRef = useRef<NodeJS.Timeout | null>(null);

  const startRecording = useCallback(async () => {
    setError(null);
    setDuration(0);
    try {
      const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
      const mr = new MediaRecorder(stream);
      chunksRef.current = [];
      mr.ondataavailable = (e) => {
        if (e.data.size > 0) chunksRef.current.push(e.data);
      };
      mr.onstop = () => {
        const blob = new Blob(chunksRef.current, { type: mr.mimeType || "audio/wav" });
        setAudioBlob(blob);
        setAudioUrl(URL.createObjectURL(blob));
        stream.getTracks().forEach((t) => t.stop());
        setState("done");
        if (timerRef.current) clearInterval(timerRef.current);
      };
      mr.start();
      mediaRef.current = mr;
      setState("recording");

      timerRef.current = setInterval(() => {
        setDuration((d) => d + 1);
      }, 1000);
    } catch (e: any) {
      setError(
        e.name === "NotAllowedError"
          ? "Microphone permission denied. Please allow access in your browser."
          : `Could not start recording: ${e.message}`
      );
    }
  }, []);

  const stopRecording = useCallback((): Promise<Blob> => {
    return new Promise((resolve) => {
      if (!mediaRef.current || mediaRef.current.state === "inactive") {
        resolve(audioBlob || new Blob([], { type: "audio/wav" }));
        return;
      }
      const existingOnStop = mediaRef.current.onstop;
      mediaRef.current.onstop = (e) => {
        if (existingOnStop) (existingOnStop as any)(e);
        const blob = new Blob(chunksRef.current, { type: "audio/wav" });
        resolve(blob);
      };
      mediaRef.current.stop();
      if (timerRef.current) clearInterval(timerRef.current);
    });
  }, [audioBlob]);

  const reset = useCallback(() => {
    if (audioUrl) URL.revokeObjectURL(audioUrl);
    setAudioUrl(null);
    setAudioBlob(null);
    setState("idle");
    setError(null);
    setDuration(0);
    if (timerRef.current) clearInterval(timerRef.current);
  }, [audioUrl]);

  useEffect(() => {
    return () => {
      if (timerRef.current) clearInterval(timerRef.current);
    };
  }, []);

  return {
    state,
    audioUrl,
    audioBlob,
    error,
    duration,
    start: startRecording,
    stop: stopRecording,
    startRecording,
    stopRecording,
    reset,
  };
}
