/**
 * src/App.tsx
 * ===========
 * Root component. Renders one of three screens based on Zustand store state:
 *
 *   store.loading = true  →  LoadingScreen  (full-page progress indicator)
 *   store.result  ≠ null  →  ResultScreen   (collage + analysis + music)
 *   otherwise             →  UploadScreen   (dropzone + vibe picker + options)
 *
 * All three screens live here so the app is runnable before the optional
 * src/components/ files are extracted.  The inner functions can be moved
 * to their own files with zero logic changes whenever you want to split them.
 *
 * CSS class names (e.g. "panel", "btn--primary") match the rules in
 * src/styles/globals.css — no inline styles for structural concerns.
 * Inline styles are used only for computed values (collage URL, seed, etc.)
 */

import React, {
  useState,
  useCallback,
  useEffect,
  useRef,
  type CSSProperties,
} from "react";
import { useDropzone, type FileRejection } from "react-dropzone";

import "@/styles/globals.css";
import { useSessionStore }                         from "@/store/sessionStore";
import { useUpload }                               from "@/hooks/useUpload";
import { useGenerate }                             from "@/hooks/useGenerate";
import type { Vibe, OutputFormat, Track }          from "@/api/types";

// ── Constants ─────────────────────────────────────────────────────────────────

const MAX_PHOTOS = 10;

const VIBES: Array<{ id: Vibe; label: string }> = [
  { id: "travel",    label: "Travel"    },
  { id: "party",     label: "Party"     },
  { id: "aesthetic", label: "Aesthetic" },
  { id: "romance",   label: "Romance"   },
  { id: "food",      label: "Food"      },
  { id: "fitness",   label: "Fitness"   },
  { id: "urban",     label: "Urban"     },
  { id: "family",    label: "Family"    },
];

const PIPELINE_STAGES = [
  "validating photos…",
  "analysing mood…",
  "selecting template…",
  "rendering collage…",
  "finding music…",
] as const;

// ── Helpers ───────────────────────────────────────────────────────────────────

function fmtMs(ms: number): string {
  const s = Math.floor(ms / 1000);
  return `${Math.floor(s / 60)}:${String(s % 60).padStart(2, "0")}`;
}

function cap(s: string): string {
  return s ? s.charAt(0).toUpperCase() + s.slice(1) : s;
}

// ── Step bar ──────────────────────────────────────────────────────────────────

const STEP_LABELS = ["upload", "generate", "preview"] as const;

function StepBar({ step }: { step: 0 | 1 | 2 }): React.JSX.Element {
  return (
    <div className="step-bar">
      {STEP_LABELS.map((label, i) => (
        <div
          key={label}
          className={[
            "step-bar__step",
            i < step  ? "step-bar__step--done"   : "",
            i === step ? "step-bar__step--active" : "",
          ].join(" ")}
        />
      ))}
      <span className="step-bar__label">{STEP_LABELS[step]}</span>
    </div>
  );
}

// ── Upload Screen ─────────────────────────────────────────────────────────────

interface FileItem {
  file:    File;
  preview: string;  // object URL — revoked on remove / unmount
}

function UploadScreen(): React.JSX.Element {
  const { vibe, format, error: storeError, setVibe, setFormat } = useSessionStore();
  const { upload, isUploading, uploadError, clearError }        = useUpload();
  const { generate, isGenerating, generateError, clearError: clearGenError } = useGenerate();

  const [items,     setItems]     = useState<FileItem[]>([]);
  const [dropError, setDropError] = useState<string | null>(null);

  // Keep a ref to the latest previews so the unmount cleanup never uses
  // a stale closure (empty dep array effect captures the initial value).
  const previewsRef = useRef<string[]>([]);
  previewsRef.current = items.map(i => i.preview);

  useEffect(() => {
    return () => {
      previewsRef.current.forEach(url => URL.revokeObjectURL(url));
    };
  }, []);

  // ── Dropzone ────────────────────────────────────────────────────────────────

  const onDrop = useCallback(
    (accepted: File[], rejected: FileRejection[]) => {
      setDropError(null);
      clearError();
      clearGenError();

      if (rejected.length > 0) {
        setDropError(
          `${rejected.length} file(s) skipped — JPEG or PNG under 20 MB only.`
        );
      }

      if (accepted.length === 0) return;

      setItems(prev => {
        const slots  = MAX_PHOTOS - prev.length;
        const toAdd  = accepted.slice(0, slots);
        const capped = accepted.length - toAdd.length;
        if (capped > 0) {
          setDropError(
            `${capped} file(s) not added — batch limit is ${MAX_PHOTOS}.`
          );
        }
        return [
          ...prev,
          ...toAdd.map(f => ({ file: f, preview: URL.createObjectURL(f) })),
        ];
      });
    },
    [clearError, clearGenError],
  );

  const { getRootProps, getInputProps, isDragActive } = useDropzone({
    onDrop,
    accept:   { "image/jpeg": [], "image/png": [] },
    maxSize:  20 * 1024 * 1024,
    multiple: true,
  });

  // ── File management ──────────────────────────────────────────────────────────

  const removeItem = useCallback((index: number) => {
    setItems(prev => {
      URL.revokeObjectURL(prev[index].preview);
      return prev.filter((_, i) => i !== index);
    });
  }, []);

  // ── Generate ─────────────────────────────────────────────────────────────────

  const handleGenerate = useCallback(async () => {
    setDropError(null);
    const ok = await upload(items.map(i => i.file));
    if (ok) await generate();
  }, [items, upload, generate]);

  const isWorking = isUploading || isGenerating;
  const topError  = dropError ?? uploadError ?? generateError ?? storeError ?? null;

  // ── Render ───────────────────────────────────────────────────────────────────

  return (
    <div className="page">
      <p className="wordmark">frame<em>it</em></p>
      <p className="tagline">AI COLLAGE GENERATOR</p>
      <StepBar step={0} />

      {topError && (
        <div className="banner banner--error">
          {topError}
        </div>
      )}

      {/* Upload panel */}
      <div className="panel">
        <p className="panel__title">Drop your photos</p>
        <p className="panel__subtitle">UP TO {MAX_PHOTOS} PHOTOS — JPEG OR PNG</p>

        <div
          {...getRootProps()}
          className={`dropzone ${isDragActive ? "dropzone--active" : ""}`}
          style={{ marginBottom: items.length > 0 ? 16 : 0 }}
        >
          <input {...getInputProps()} />
          <svg
            className="dropzone__icon"
            viewBox="0 0 40 40" fill="none"
            xmlns="http://www.w3.org/2000/svg"
          >
            <rect x="6" y="12" width="28" height="20" rx="3"
                  stroke="currentColor" strokeWidth="1.5"/>
            <circle cx="15" cy="21" r="3"
                    stroke="currentColor" strokeWidth="1.5"/>
            <path d="M6 26l8-6 6 5 5-4 9 7"
                  stroke="currentColor" strokeWidth="1.5"
                  strokeLinecap="round" strokeLinejoin="round"/>
            <path d="M20 6v9M17 9l3-3 3 3"
                  stroke="currentColor" strokeWidth="1.5"
                  strokeLinecap="round" strokeLinejoin="round"/>
          </svg>
          <p className="dropzone__text">
            {isDragActive
              ? "Drop photos here"
              : items.length === 0
              ? "Click or drag photos here"
              : "Add more photos"}
          </p>
          <p className="dropzone__count">{items.length}/{MAX_PHOTOS} selected</p>
        </div>

        {items.length > 0 && (
          <div className="thumb-grid">
            {items.map((item, idx) => (
              <div key={item.preview} className="thumb">
                <img src={item.preview} alt={`Photo ${idx + 1}`} />
                <button
                  className="thumb__remove"
                  onClick={e => { e.stopPropagation(); removeItem(idx); }}
                  aria-label={`Remove photo ${idx + 1}`}
                >
                  ×
                </button>
              </div>
            ))}
          </div>
        )}
      </div>

      {/* Options panel */}
      <div className="panel">
        <p className="panel__title" style={{ fontSize: 18 }}>Vibe</p>
        <p className="panel__subtitle">OPTIONAL — AUTO-DETECTED FROM YOUR PHOTOS</p>

        <div className="vibe-row">
          {VIBES.map(v => (
            <button
              key={v.id}
              onClick={() => setVibe(vibe === v.id ? null : v.id)}
              className={`vibe-pill ${vibe === v.id ? "vibe-pill--selected" : ""}`}
            >
              {v.label}
            </button>
          ))}
        </div>

        <p className="panel__subtitle" style={{ marginBottom: 12 }}>FORMAT</p>
        <div className="format-row">
          {(["post", "story"] as OutputFormat[]).map(f => (
            <button
              key={f}
              onClick={() => setFormat(f)}
              className={`format-btn ${format === f ? "format-btn--selected" : ""}`}
            >
              {f === "post" ? "Post" : "Story"}
              <span className="format-btn__dim">
                {f === "post" ? "1080 × 1080" : "1080 × 1920"}
              </span>
            </button>
          ))}
        </div>

        <button
          className="btn btn--primary"
          onClick={() => void handleGenerate()}
          disabled={items.length === 0 || isWorking}
        >
          {isUploading
            ? "Uploading…"
            : isGenerating
            ? "Creating collage…"
            : items.length === 0
            ? "Select photos to continue"
            : "Generate collage  →"}
        </button>
      </div>
    </div>
  );
}

// ── Loading Screen ─────────────────────────────────────────────────────────────

function LoadingScreen(): React.JSX.Element {
  const [progress, setProgress] = useState(0);
  const [stage,    setStage]    = useState(PIPELINE_STAGES[0]);

  useEffect(() => {
    const interval = setInterval(() => {
      setProgress(prev => {
        const next   = Math.min(100, prev + 1.8);
        const idx    = Math.min(
          Math.floor((next / 100) * PIPELINE_STAGES.length),
          PIPELINE_STAGES.length - 1
        );
        setStage(PIPELINE_STAGES[idx]);
        if (next >= 100) clearInterval(interval);
        return next;
      });
    }, 75);
    return () => clearInterval(interval);
  }, []);

  return (
    <div className="page loading-page">
      <p className="wordmark" style={{ fontSize: 26, textAlign: "center", marginBottom: 28 }}>
        frame<em>it</em>
      </p>

      <div className="spinner" />

      <div className="progress-bar-wrap">
        <div className="progress-bar-track">
          <div
            className="progress-bar-fill"
            style={{ width: `${progress}%` }}
          />
        </div>
        <p className="progress-label">{stage.toUpperCase()}</p>
      </div>
    </div>
  );
}

// ── Result Screen ──────────────────────────────────────────────────────────────

function ResultScreen(): React.JSX.Element {
  const { result, reset }          = useSessionStore();
  const { generate, isGenerating } = useGenerate();

  const [imgLoaded,    setImgLoaded]    = useState(false);
  const [hideWarnings, setHideWarnings] = useState(false);
  const [imgError,     setImgError]     = useState(false);

  // Safety net: if result is somehow null, render upload instead of crashing
  if (!result) return <UploadScreen />;

  const { music_tracks: tracks, warnings, format } = result;
  const isStory = format === "story";

  const analysisRows: Array<[string, string]> = [
    ["mood",        cap(result.detected_mood)],
    ["orientation", result.detected_orientation],
    ["energy",      result.energy],
    ["template",    result.selected_template.replace(/_/g, " ")],
  ];

  return (
    <div className="page animate-fade-up">

      <p className="wordmark" style={{ marginBottom: 20 }}>frame<em>it</em></p>

      {/* Warnings banner */}
      {warnings.length > 0 && !hideWarnings && (
        <div className="banner banner--warning">
          <div className="banner__text">
            {warnings.map((w, i) => (
              <div key={i}>⚠ {w}</div>
            ))}
          </div>
          <button
            className="banner__close"
            onClick={() => setHideWarnings(true)}
            aria-label="Dismiss warnings"
          >×</button>
        </div>
      )}

      <div className="result-grid">

        {/* Left: collage + actions */}
        <div>
          <div className={`collage-wrap ${isStory ? "collage-wrap--story" : ""}`}>
            {/* Loading spinner while image loads */}
            {!imgLoaded && !imgError && (
              <div className="collage-wrap__loading">
                <div className="spinner" style={{ width: 32, height: 32,
                  borderColor: "rgba(255,255,255,0.15)",
                  borderTopColor: "rgba(255,255,255,0.65)" }}
                />
              </div>
            )}

            {/* Error fallback */}
            {imgError && (
              <div className="collage-wrap__loading" style={{
                flexDirection: "column", gap: 8, color: "rgba(255,255,255,0.4)",
                fontFamily: "var(--font-mono)", fontSize: 11, textAlign: "center", padding: 20,
              }}>
                <span style={{ fontSize: 32 }}>🖼</span>
                <span>Preview unavailable</span>
              </div>
            )}

            {/* Collage image */}
            <img
              src={result.collage_url}
              alt="Generated collage"
              onLoad={() => setImgLoaded(true)}
              onError={() => { setImgError(true); setImgLoaded(false); }}
              style={{ display: imgLoaded ? "block" : "none" }}
            />

            <div className="seed-badge">seed {result.seed}</div>
          </div>

          {/* Actions */}
          <div className="action-row">
            <button
              className="btn btn--secondary"
              onClick={() => void generate()}
              disabled={isGenerating}
              style={{ opacity: isGenerating ? 0.45 : 1,
                       cursor: isGenerating ? "not-allowed" : "pointer" }}
            >
              {isGenerating ? "Generating…" : "↻  Regenerate"}
            </button>

            <a
              className="btn btn--download"
              href={`/api/v1/download/${result.collage_filename}`}
              download={result.collage_filename}
              style={{ flex: "1.4", textDecoration: "none" }}
            >
              ↓  Download PNG
            </a>
          </div>
        </div>

        {/* Right: analysis + music */}
        <div className="right-col">

          {/* Analysis */}
          <div className="analysis-card">
            <p className="card-label">ANALYSIS</p>

            {analysisRows.map(([key, val]) => (
              <div key={key} className="analysis-row">
                <span className="analysis-key">{key}</span>
                <span className="analysis-val">{val}</span>
              </div>
            ))}

            {/* Scene tags + palette */}
            <div className="tags-row">
              {result.scene_tags.map(tag => (
                <span key={tag} className="tag">{tag}</span>
              ))}
              {result.palette_hex.slice(0, 5).map(hex => (
                <span
                  key={hex}
                  className="palette-dot"
                  style={{ background: hex }}
                  title={hex}
                />
              ))}
            </div>
          </div>

          {/* Music */}
          {tracks.length > 0 ? (
            <div className="music-card">
              <p className="card-label">RECOMMENDED TRACKS</p>
              <div className="music-scroll">
                {tracks.slice(0, 6).map((track: Track, i: number) => (
                  <div key={track.id} className="track">
                    <span className="track__num">{i + 1}</span>

                    <div className="track__info">
                      <p className="track__name">{track.name}</p>
                      <p className="track__artist">{track.artist}</p>
                    </div>

                    {track.duration_ms > 0 && (
                      <span className="track__dur">{fmtMs(track.duration_ms)}</span>
                    )}

                    {(track.preview_url ?? track.spotify_url) && (
                      <a
                        href={track.preview_url ?? track.spotify_url}
                        target="_blank"
                        rel="noopener noreferrer"
                        className="track__play"
                        aria-label={`Play ${track.name}`}
                      >▶</a>
                    )}
                  </div>
                ))}
              </div>
            </div>
          ) : (
            <div className="music-card">
              <p className="card-label">RECOMMENDED TRACKS</p>
              <p className="music-empty">
                Music recommendations unavailable.
                <br />
                Add Spotify credentials to .env to enable.
              </p>
            </div>
          )}
        </div>
      </div>

      {/* Timing row (dev-mode detail) */}
      {import.meta.env.DEV && (
        <div style={{
          fontFamily: "var(--font-mono)", fontSize: 10, color: "var(--warm)",
          textAlign: "center", marginBottom: 12, letterSpacing: "0.04em",
        }}>
          {Object.entries(result.timing_ms).map(([k, v]) => (
            <span key={k} style={{ marginRight: 12 }}>{k}: {v}ms</span>
          ))}
        </div>
      )}

      {/* Start over */}
      <div className="start-over">
        <button onClick={() => reset()}>START OVER</button>
      </div>
    </div>
  );
}

// ── Root ──────────────────────────────────────────────────────────────────────

export default function App(): React.JSX.Element {
  const { result, loading } = useSessionStore();

  if (loading) return <LoadingScreen />;
  if (result)  return <ResultScreen />;
  return <UploadScreen />;
}