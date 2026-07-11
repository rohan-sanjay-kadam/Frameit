/**
 * src/hooks/useGenerate.ts
 * ========================
 * Sends POST /api/v1/generate and drives screen switching via the Zustand store.
 *
 * Screen transitions
 * ------------------
 *   generate() called     →  store.loading = true   →  App renders LoadingScreen
 *   request succeeds      →  store.setResult(data)  →  App renders ResultScreen
 *   request fails         →  store.loading = false  →  App renders UploadScreen
 *                                                        with generateError shown
 *
 * Stale closure prevention
 * ------------------------
 * We read sessionId, photoIds, vibe, and format from useSessionStore.getState()
 * inside the generate() callback rather than from the render-time closure.
 *
 * Why this matters: useUpload calls setSession() right before useGenerate
 * is called in the same event handler.  Zustand updates are synchronous, so
 * getState() always returns the freshest values — the closure from the last
 * render would give us the stale pre-upload values (sessionId = null).
 *
 * Regenerate
 * ----------
 * Calling generate() with an explicit seed replays the same layout.
 * Calling generate() without a seed (or seed=undefined) lets the server
 * pick a new random seed — produces a different layout with the same photos.
 *
 * Returns
 * -------
 *   generate(seed?)   Async. Does nothing if sessionId is null.
 *   isGenerating      True while the request is in-flight.
 *   generateError     String message on failure, null otherwise.
 *   clearError        Resets generateError to null.
 */

import { useState, useCallback } from "react";
import { useSessionStore }        from "@/store/sessionStore";
import type { GenerateResponse, ApiErrorDetail, ApiErrorResponse } from "@/api/types";

// ── Hook ─────────────────────────────────────────────────────────────────────

export interface UseGenerateReturn {
  generate:      (seed?: number) => Promise<void>;
  isGenerating:  boolean;
  generateError: string | null;
  clearError:    () => void;
}

export function useGenerate(): UseGenerateReturn {
  // Subscribe only to the setters — avoids re-renders when state values change
  const setResult  = useSessionStore(s => s.setResult);
  const setLoading = useSessionStore(s => s.setLoading);
  const setError   = useSessionStore(s => s.setError);

  const [isGenerating,  setIsGenerating]  = useState(false);
  const [generateError, setGenerateError] = useState<string | null>(null);

  const generate = useCallback(async (seed?: number): Promise<void> => {
    // Read fresh state at call time — avoids stale closure after useUpload updates
    const { sessionId, photoIds, vibe, format } = useSessionStore.getState();

    if (!sessionId || photoIds.length === 0) {
      setGenerateError("No uploaded photos found. Please upload photos first.");
      return;
    }

    // Set both local + store loading states before the first await so there is
    // no frame where the UI is between "working" states.
    setIsGenerating(true);
    setLoading(true);
    setGenerateError(null);
    setError(null);

    try {
      const requestBody = {
        session_id: sessionId,
        photo_ids:  photoIds,
        format,
        ...(vibe  !== null      ? { vibe }  : {}),
        ...(seed  !== undefined ? { seed }  : {}),
      };

      const res = await fetch("/api/v1/generate", {
        method:  "POST",
        headers: { "Content-Type": "application/json" },
        body:    JSON.stringify(requestBody),
      });

      if (!res.ok) {
        // The server may return either a string detail or an object detail
        // (see api/routes/generate.py for the 500 shape)
        const errBody = await res.json().catch(
          (): ApiErrorResponse => ({ error: `Server error (HTTP ${res.status})` })
        ) as ApiErrorResponse;

        const detail = errBody.detail;
        const message =
          typeof detail === "object" && detail !== null
            ? (detail as ApiErrorDetail).message
            : (detail as string | undefined)
              ?? errBody.error
              ?? `Generation failed (HTTP ${res.status})`;

        throw new Error(message);
      }

      const data: GenerateResponse = await res.json();

      // setResult atomically sets result + loading=false (one Zustand update)
      // so App.tsx transitions LoadingScreen → ResultScreen in a single render.
      setResult(data);
    } catch (err) {
      const message =
        err instanceof Error
          ? err.message
          : "Collage generation failed. Please try again.";

      setGenerateError(message);
      setError(message);
      setLoading(false);   // return to UploadScreen
    } finally {
      setIsGenerating(false);
    }
  }, [setResult, setLoading, setError]);

  const clearError = useCallback(() => setGenerateError(null), []);

  return { generate, isGenerating, generateError, clearError };
}