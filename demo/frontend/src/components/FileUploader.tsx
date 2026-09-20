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
      className={`dropzone${drag ? " drag-over" : ""}`}
      onClick={() => inputRef.current?.click()}
      onDragOver={(e) => { e.preventDefault(); setDrag(true); }}
      onDragLeave={() => setDrag(false)}
      onDrop={(e) => {
        e.preventDefault();
        setDrag(false);
        const f = e.dataTransfer.files[0];
        if (f) handle(f);
      }}
    >
      <p>Drop an audio file here, or click to browse</p>
      <small>WAV, MP3, FLAC, OGG, M4A, WEBM</small>
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
