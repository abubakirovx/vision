import { useCallback, useEffect, useRef, useState } from "react";

export function useCamera() {
  const videoRef = useRef<HTMLVideoElement>(null);
  const streamRef = useRef<MediaStream | null>(null);
  const generation = useRef(0);
  const canvasRef = useRef<HTMLCanvasElement | null>(null);
  const [active, setActive] = useState(false);
  const [starting, setStarting] = useState(false);
  const [error, setError] = useState("");
  const [resolution, setResolution] = useState("");
  const [dimensions, setDimensions] = useState({ width: 1280, height: 720 });
  const [session, setSession] = useState(0);

  const stop = useCallback(() => {
    generation.current++;
    streamRef.current?.getTracks().forEach((track) => track.stop());
    streamRef.current = null;
    if (videoRef.current) {
      videoRef.current.pause();
      videoRef.current.onresize = null;
      videoRef.current.srcObject = null;
    }
    if (canvasRef.current)
      canvasRef.current.width = canvasRef.current.height = 0;
    setActive(false);
    setStarting(false);
    setResolution("");
    setSession((value) => value + 1);
  }, []);

  const start = useCallback(async () => {
    stop();
    const token = generation.current;
    setStarting(true);
    setError("");
    try {
      if (!navigator.mediaDevices?.getUserMedia)
        throw new Error(
          "Camera requires localhost or HTTPS and a supported browser.",
        );
      const stream = await navigator.mediaDevices.getUserMedia({
        video: {
          width: { ideal: 1280 },
          height: { ideal: 720 },
          facingMode: "environment",
        },
        audio: false,
      });
      if (token !== generation.current) {
        stream.getTracks().forEach((track) => track.stop());
        return;
      }
      streamRef.current = stream;
      const video = videoRef.current!;
      video.srcObject = stream;
      stream.getVideoTracks()[0].addEventListener(
        "ended",
        () => {
          if (token !== generation.current) return;
          stop();
          setError("Camera was disconnected. Reconnect it and start again.");
        },
        { once: true },
      );
      video.onresize = () => {
        if (
          token !== generation.current ||
          !video.videoWidth ||
          !video.videoHeight
        )
          return;
        setDimensions({ width: video.videoWidth, height: video.videoHeight });
        setResolution(`${video.videoWidth} × ${video.videoHeight}`);
      };
      await video.play();
      if (token !== generation.current) return;
      if (!stream.active)
        throw new Error(
          "Camera disconnected while starting. Reconnect it and try again.",
        );
      setDimensions({ width: video.videoWidth, height: video.videoHeight });
      setResolution(`${video.videoWidth} × ${video.videoHeight}`);
      setActive(true);
    } catch (err) {
      if (token !== generation.current) return;
      stop();
      const name = err instanceof DOMException ? err.name : "";
      setError(
        name === "NotAllowedError"
          ? "Camera access denied. Allow camera permission in your browser and try again."
          : name === "NotFoundError"
            ? "No camera found. Connect a webcam and try again."
            : name === "NotReadableError"
              ? "Camera is in use by another app. Close it and try again."
              : err instanceof Error
                ? err.message
                : "Could not start the camera.",
      );
    } finally {
      if (token === generation.current) setStarting(false);
    }
  }, [stop]);

  const capture = useCallback((quality = 0.85) => {
    const video = videoRef.current;
    if (
      !streamRef.current?.active ||
      !video ||
      video.readyState < 2 ||
      !video.videoWidth
    )
      throw new Error("Start the camera before capturing a frame.");
    const scale = Math.min(
      1,
      1280 / Math.max(video.videoWidth, video.videoHeight),
    );
    const canvas = (canvasRef.current ??= document.createElement("canvas"));
    canvas.width = Math.round(video.videoWidth * scale);
    canvas.height = Math.round(video.videoHeight * scale);
    const context = canvas.getContext("2d");
    if (!context)
      throw new Error(
        "This browser could not capture a frame. Try another browser.",
      );
    context.drawImage(video, 0, 0, canvas.width, canvas.height);
    return canvas.toDataURL("image/jpeg", quality);
  }, []);

  useEffect(() => {
    const video = videoRef.current;
    function release() {
      generation.current++;
      streamRef.current?.getTracks().forEach((track) => track.stop());
      streamRef.current = null;
      canvasRef.current = null;
      if (video) {
        video.onresize = null;
        video.srcObject = null;
      }
    }
    window.addEventListener("pagehide", stop);
    return () => {
      window.removeEventListener("pagehide", stop);
      release();
    };
  }, [stop]);
  return {
    videoRef,
    active,
    starting,
    error,
    resolution,
    dimensions,
    session,
    start,
    stop,
    capture,
  };
}
