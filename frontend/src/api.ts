import type { DetectionResult, Health } from "./types";

const base = (import.meta.env.VITE_API_URL || "").replace(/\/$/, "");

async function request<T>(
  path: string,
  options: RequestInit = {},
  timeout = 15000,
): Promise<T> {
  const controller = new AbortController();
  const abort = () => controller.abort();
  options.signal?.addEventListener("abort", abort, { once: true });
  if (options.signal?.aborted) controller.abort();
  const timer = window.setTimeout(abort, timeout);
  try {
    const response = await fetch(`${base}${path}`, {
      ...options,
      signal: controller.signal,
    });
    const data = await response.json().catch(() => null);
    if (!response.ok) {
      throw new Error(
        typeof data?.detail === "string"
          ? data.detail
          : `Request failed (${response.status}). Check the backend.`,
      );
    }
    if (!data)
      throw new Error(
        "Invalid backend response. Check that FastAPI is running on port 8000.",
      );
    return data as T;
  } catch (error) {
    if (controller.signal.aborted)
      throw new Error(
        options.signal?.aborted
          ? "Request cancelled."
          : "The request timed out. The model may still be loading; try again shortly.",
      );
    if (error instanceof TypeError)
      throw new Error("Cannot reach the backend. Start FastAPI on port 8000.");
    throw error;
  } finally {
    window.clearTimeout(timer);
    options.signal?.removeEventListener("abort", abort);
  }
}

export const getHealth = (signal?: AbortSignal) =>
  request<Health>("/health", { signal }, 6000);
export const detectFrame = async (image: string, signal: AbortSignal) => {
  const started = performance.now();
  const result = await request<DetectionResult>(
    "/api/detect",
    {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ image }),
      signal,
    },
    15000,
  );
  return { ...result, roundtrip_ms: Math.round(performance.now() - started) };
};
export const askFrame = (
  image: string,
  question: string,
  language: string,
  signal: AbortSignal,
) =>
  request<{ answer: string; elapsed_ms: number }>(
    "/api/ask",
    {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ image, question, language }),
      signal,
    },
    195000,
  );
