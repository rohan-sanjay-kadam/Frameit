/**
 * src/store/sessionStore.ts
 * =========================
 * Zustand store — the single source of truth for the entire application.
 *
 * State drives screen selection in App.tsx:
 *   loading = true   →  LoadingScreen
 *   result  ≠ null   →  ResultScreen
 *   otherwise        →  UploadScreen
 *
 * Design notes
 * ------------
 * - All state mutations go through explicit action functions.
 *   No direct `set({ ... })` outside this file.
 * - `setResult` atomically sets both result and loading=false so there is
 *   never a frame where result is set but the loading screen is still shown.
 * - `reset` returns to the exact initial state without a page reload.
 * - `getState()` is used by hooks that need fresh values at call time
 *   rather than at render time (avoids stale closures across async boundaries).
 */

import { create } from "zustand";
import type { GenerateResponse, OutputFormat, Vibe } from "@/api/types";

// ── State shape ───────────────────────────────────────────────────────────────

interface SessionState {
  // ── Upload phase ─────────────────────────────────────────────────────────
  sessionId: string | null;   // UUID returned by /upload; null until first upload
  photoIds:  string[];        // filenames on the server, aligned with uploaded files

  // ── User options ─────────────────────────────────────────────────────────
  vibe:   Vibe | null;        // null = auto-detect from image analysis
  format: OutputFormat;       // "post" | "story"

  // ── Generate phase ───────────────────────────────────────────────────────
  result:  GenerateResponse | null;
  loading: boolean;           // true while generate is in-flight → shows LoadingScreen
  error:   string | null;     // last error message; cleared at the start of each generate

  // ── Actions ──────────────────────────────────────────────────────────────
  setSession: (sessionId: string, photoIds: string[]) => void;
  setVibe:    (vibe: Vibe | null) => void;
  setFormat:  (format: OutputFormat) => void;

  /**
   * Called by useGenerate on a successful response.
   * Atomically updates result and clears loading so App.tsx transitions
   * from LoadingScreen → ResultScreen in a single render.
   */
  setResult:  (result: GenerateResponse) => void;

  setLoading: (loading: boolean)        => void;
  setError:   (error: string | null)    => void;

  /** Return to the exact initial state. Called by "Start over" in ResultScreen. */
  reset: () => void;
}

// ── Initial state ─────────────────────────────────────────────────────────────

const INITIAL: Omit<SessionState, keyof { [K in keyof SessionState as SessionState[K] extends Function ? K : never]: unknown }> = {
  sessionId: null,
  photoIds:  [],
  vibe:      null,
  format:    "post",
  result:    null,
  loading:   false,
  error:     null,
};

// Simpler constant — just the data fields:
const INITIAL_DATA = {
  sessionId: null  as string | null,
  photoIds:  []    as string[],
  vibe:      null  as Vibe | null,
  format:    "post" as OutputFormat,
  result:    null  as GenerateResponse | null,
  loading:   false as boolean,
  error:     null  as string | null,
};

// ── Store ─────────────────────────────────────────────────────────────────────

export const useSessionStore = create<SessionState>()((set) => ({
  ...INITIAL_DATA,

  setSession: (sessionId, photoIds) =>
    set({ sessionId, photoIds }),

  setVibe:   (vibe)   => set({ vibe }),
  setFormat: (format) => set({ format }),

  // Atomic: clears loading AND sets result in one update so App.tsx
  // never renders with (loading=true, result=<value>) simultaneously.
  setResult: (result) => set({ result, loading: false, error: null }),

  setLoading: (loading) => set({ loading }),
  setError:   (error)   => set({ error }),

  reset: () => set({ ...INITIAL_DATA }),
}));