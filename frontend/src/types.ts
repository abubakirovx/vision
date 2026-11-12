export type Detection = {
  label: string;
  confidence: number;
  box: [number, number, number, number];
};
export type DetectionResult = {
  objects: Detection[];
  counts: Record<string, number>;
  total: number;
  inference_ms: number;
  processing_ms?: number;
  roundtrip_ms?: number;
  width: number;
  height: number;
};
export type Health = {
  status: string;
  detector: {
    status: "ready" | "loading" | "error";
    model: string;
    error: string | null;
    image_size: number;
    confidence: number;
    device: string;
  };
  ollama: {
    status: "ready" | "offline" | "model_missing";
    model: string;
    busy?: boolean;
  };
};
export type Message = {
  id: number;
  role: "user" | "assistant";
  text: string;
  time: string;
  elapsed?: number;
  error?: boolean;
  language?: "en" | "uz";
};
