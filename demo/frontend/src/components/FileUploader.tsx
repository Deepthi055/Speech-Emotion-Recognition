"use client";

import { useRef, useState, useCallback } from "react";

interface Props {
  onFile: (file: File, url: string) => void;
}

export function FileUploader({ onFile }: Props) {
  const [drag, setDrag] = useState(false);
  const inputRef = useRef<HTMLInputElement>(null);

  const handle = useCallback(
    (file: File) => {
      const url = URL.createObjectURL(file);
      onFile(file, url);
    },
    [onFile]
  );

  return (
    <div
      onClick={() => inputRef.current?.click()}
      onDragOver={(e) => {
        e.preventDefault();
        setDrag(true);
      }}
      onDragLeave={() => setDrag(false)}
      onDrop={(e) => {
        e.preventDefault();
        setDrag(false);
        const f = e.dataTransfer.files[0];
        if (f) handle(f);
      }}
      style={{
        border: `1px dashed ${drag ? "var(--text-primary)" : "var(--border)"}`,
        borderRadius: "3px",
        padding: "2rem 1.25rem",
        textAlign: "center",
        cursor: "pointer",
        background: drag ? "var(--bg-element)" : "var(--bg-subtle)",
        transition: "border-color 0.12s, background-color 0.12s",
        display: "flex",
        flexDirection: "column",
        alignItems: "center",
        justifyContent: "center",
        gap: "0.6rem",
      }}
    >
      <div
        style={{
          width: "32px",
          height: "32px",
          borderRadius: "2px",
          background: "var(--bg-panel)",
          border: "1px solid var(--border)",
          display: "flex",
          alignItems: "center",
          justifyContent: "center",
          color: "var(--text-secondary)",
        }}
      >
        <svg width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.5">
          <path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4" />
          <polyline points="17 8 12 3 7 8" />
          <line x1="12" y1="3" x2="12" y2="15" />
        </svg>
      </div>

      <div>
        <p style={{ fontSize: "0.85rem", fontWeight: 500, color: "var(--text-primary)" }}>
          Select or drop audio payload
        </p>
        <p className="font-mono" style={{ fontSize: "0.7rem", color: "var(--text-muted)", marginTop: "2px" }}>
          PCM / WAV, MP3, FLAC, OGG, M4A, WEBM
        </p>
      </div>

      <input
        ref={inputRef}
        type="file"
        accept="audio/*"
        style={{ display: "none" }}
        onChange={(e) => {
          const f = e.target.files?.[0];
          if (f) handle(f);
          e.target.value = "";
        }}
      />
    </div>
  );
}
