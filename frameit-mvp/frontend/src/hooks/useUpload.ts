/**
 * src/hooks/useUpload.ts
 * ======================
 * Validates a list of File objects and POSTs them to POST /api/v1/upload.
 *
 * On success: updates the Zustand store with session_id and photo_ids so
 * useGenerate can read them at call time.
 *
 * On failure: sets a local error string and returns false — the caller
 * (App.tsx's handleGenerate) aborts the generate flow immediately.
 *
 * Why not set store.loading here?
 * --------------------------------
 * Upload is synchronous from the user's perspective — the UploadScreen
 * stays visible with the button showing "Uploading…".  Only the generate
 * phase switches to the full LoadingScreen.  This gives the user better
 * feedback during the two-phase flow.
 *
 * Client-side pre-validation
 * --------------------------
 * We mirror the server's checks (type, size) so errors surface instantly
 * without a network round-trip.  The server validates again independently.
 *
 * Content-Type header
 * -------------------
 * Do NOT set Content-Type manually when using FormData.  The browser must
 * set it so it can include the multipart boundary string.  Any manual
 * Content-Type header breaks the server's file parsing.
 */

import { useState, useCallback } from "react";
import { useSessionStore }       from "@/store/sessionStore";
import type { UploadResponse }   from "@/api/types";

// ── Constants (mirrors server-side limits in api/routes/upload.py) ────────────

const ALLOWED_TYPES  = new Set(["image/jpeg", "image/jpg", "image/png"]);
const MAX_BYTES      = 20 * 1024 * 1024;   // 20 MB
const MAX_PHOTOS     = 10;

// ── Hook ─────────────────────────────────────────────────────────────────────

export interface UseUploadReturn {
  /**
   * Upload files to the server.
   * Returns true on success (store is updated), false on any error.
   */
  upload:       (files: File[]) => Promise<boolean>;
  isUploading:  boolean;
  uploadError:  string | null;
  clearError:   () => void;
}

export function useUpload(): UseUploadReturn {
  const setSession = useSessionStore(s => s.setSession);

  const [isUploading, setIsUploading] = useState(false);
  const [uploadError, setUploadError] = useState<string | null>(null);

  const upload = useCallback(async (files: File[]): Promise<boolean> => {
    // ── Client-side pre-validation ──────────────────────────────────────────
    if (files.length === 0) {
      setUploadError("No files selected. Please choose at least one photo.");
      return false;
    }

    const valid   = files.filter(f => ALLOWED_TYPES.has(f.type) && f.size <= MAX_BYTES);
    const invalid = files.length - valid.length;

    if (valid.length === 0) {
      setUploadError("No valid images found. Use JPEG or PNG files under 20 MB.");
      return false;
    }

    const capped    = valid.slice(0, MAX_PHOTOS);
    const truncated = valid.length - capped.length;

    // ── Upload ────────────────────────────────────────────────────────────────
    setIsUploading(true);
    setUploadError(null);

    try {
      const formData = new FormData();
      // Field name "files" matches the FastAPI route parameter name
      capped.forEach(f => formData.append("files", f));

      const res = await fetch("/api/v1/upload", {
        method: "POST",
        body:   formData,
        // Do NOT set Content-Type — the browser sets the multipart boundary
      });

      if (!res.ok) {
        const body = await res.json().catch(() => null) as Record<string, unknown> | null;
        const detail = body?.detail;
        const message =
          typeof detail === "string"
            ? detail
            : (body?.error as string | undefined)
              ?? `Upload failed (HTTP ${res.status})`;
        throw new Error(message);
      }

      const data: UploadResponse = await res.json();

      // Guard: server rejected everything
      if (data.accepted === 0) {
        throw new Error(
          data.warnings.length > 0
            ? data.warnings.join(" ")
            : "All files were rejected by the server. Check file type and size."
        );
      }

      // Persist session to store so useGenerate can read at call time
      setSession(data.session_id, data.photo_ids);

      // Surface informational notices (not errors) to the caller via
      // uploadError so the UploadScreen can show them as chips
      const notices: string[] = [];
      if (invalid > 0)    notices.push(`${invalid} file(s) skipped (wrong type or size).`);
      if (truncated > 0)  notices.push(`${truncated} file(s) dropped — limit is ${MAX_PHOTOS}.`);
      if (data.rejected > 0) notices.push(...data.warnings);
      if (notices.length > 0) setUploadError(notices.join(" "));

      return true;
    } catch (err) {
      setUploadError(
        err instanceof Error
          ? err.message
          : "Upload failed. Please try again."
      );
      return false;
    } finally {
      setIsUploading(false);
    }
  }, [setSession]);

  const clearError = useCallback(() => setUploadError(null), []);

  return { upload, isUploading, uploadError, clearError };
}