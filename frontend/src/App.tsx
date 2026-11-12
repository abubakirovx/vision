import { useCallback, useEffect, useRef, useState } from "react";
import {
  ArrowUp,
  Camera,
  Grid2X2,
  LoaderCircle,
  Maximize2,
  MessageSquare,
  Settings2,
  Square,
  Trash2,
  X,
} from "lucide-react";
import { askFrame, detectFrame, getHealth } from "./api";
import { useCamera } from "./useCamera";
import DetectionOverlay from "./DetectionOverlay";
import CameraStage from "./CameraStage";
import type { DetectionResult, Health, Message } from "./types";

const emptyResult: DetectionResult = {
  objects: [],
  counts: {},
  total: 0,
  inference_ms: 0,
  width: 0,
  height: 0,
};
const timeNow = () =>
  new Date().toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" });

export default function App() {
  const camera = useCamera();
  const [health, setHealth] = useState<Health | null>(null);
  const [healthChecked, setHealthChecked] = useState(false);
  const [backendError, setBackendError] = useState("");
  const [detectionError, setDetectionError] = useState("");
  const [result, setResult] = useState<DetectionResult>(emptyResult);
  const [frames, setFrames] = useState(0);
  const [detecting, setDetecting] = useState(false);
  const [showBoxes, setShowBoxes] = useState(true);
  const [language, setLanguage] = useState<"en" | "uz">("en");
  const [question, setQuestion] = useState("");
  const [messages, setMessages] = useState<Message[]>([]);
  const [asking, setAsking] = useState(false);
  const [askError, setAskError] = useState("");
  const [pageVisible, setPageVisible] = useState(!document.hidden);
  const [askSeconds, setAskSeconds] = useState(0);
  const [lastDetectionAt, setLastDetectionAt] = useState(0);
  const [now, setNow] = useState(performance.now());
  const [help, setHelp] = useState(false);
  const askController = useRef<AbortController | null>(null);
  const chatEnd = useRef<HTMLDivElement>(null);
  const cameraPanel = useRef<HTMLDivElement>(null);
  const chatInput = useRef<HTMLTextAreaElement>(null);
  const messageId = useRef(0);
  const askBusy = useRef(false);
  const askStarted = useRef(0);
  const mounted = useRef(true);
  const [view, setView] = useState<"workspace" | "chat">("workspace");

  useEffect(() => {
    const controller = new AbortController();
    let timer: number;
    async function poll() {
      try {
        const next = await getHealth(controller.signal);
        if (controller.signal.aborted) return;
        setHealth(next);
        setBackendError("");
      } catch (err) {
        if (controller.signal.aborted) return;
        setHealth(null);
        setBackendError(
          err instanceof Error ? err.message : "Backend is offline.",
        );
      }
      if (!controller.signal.aborted) {
        setHealthChecked(true);
        timer = window.setTimeout(poll, 10000);
      }
    }
    void poll();
    return () => {
      controller.abort();
      window.clearTimeout(timer);
    };
  }, []);

  useEffect(() => {
    setResult(emptyResult);
    setDetectionError("");
    if (!camera.active || !pageVisible) {
      setResult(emptyResult);
      setDetecting(false);
      setDetectionError("");
      return;
    }
    const controller = new AbortController();
    let timer: number;
    async function detect() {
      const started = performance.now();
      setDetecting(true);
      try {
        const next = await detectFrame(camera.capture(), controller.signal);
        if (controller.signal.aborted) return;
        setResult(next);
        setLastDetectionAt(performance.now());
        setFrames((count) => count + 1);
        setDetectionError("");
      } catch (err) {
        if (controller.signal.aborted) return;
        setDetectionError(
          err instanceof Error ? err.message : "Detection failed.",
        );
        setResult(emptyResult);
      } finally {
        if (!controller.signal.aborted) {
          setDetecting(false);
          timer = window.setTimeout(
            detect,
            Math.max(
              500,
              (document.hidden ? 10000 : 1500) - (performance.now() - started),
            ),
          );
        }
      }
    }
    void detect();
    return () => {
      controller.abort();
      window.clearTimeout(timer);
    };
  }, [
    camera.active,
    camera.capture,
    camera.session,
    camera.dimensions.width,
    camera.dimensions.height,
    pageVisible,
  ]);

  useEffect(() => {
    const changed = () => setPageVisible(!document.hidden);
    document.addEventListener("visibilitychange", changed);
    return () => document.removeEventListener("visibilitychange", changed);
  }, []);

  useEffect(() => {
    if (!camera.active && !asking) return;
    const timer = window.setInterval(() => {
      setNow(performance.now());
      if (asking)
        setAskSeconds(
          Math.floor((performance.now() - askStarted.current) / 1000),
        );
    }, 1000);
    return () => window.clearInterval(timer);
  }, [camera.active, asking]);

  useEffect(() => {
    chatEnd.current?.scrollIntoView({ behavior: "smooth", block: "nearest" });
  }, [messages, asking]);
  useEffect(() => {
    mounted.current = true;
    return () => {
      mounted.current = false;
      askController.current?.abort();
    };
  }, []);

  useEffect(() => {
    if (!help) return;
    const previous = document.activeElement as HTMLElement | null;
    const overflow = document.body.style.overflow;
    document.body.style.overflow = "hidden";
    function trapFocus(event: KeyboardEvent) {
      if (event.key === "Escape") setHelp(false);
      if (event.key !== "Tab") return;
      const controls =
        document.querySelectorAll<HTMLElement>(".help-modal button");
      const first = controls[0];
      const last = controls[controls.length - 1];
      if (event.shiftKey && document.activeElement === first) {
        event.preventDefault();
        last?.focus();
      } else if (!event.shiftKey && document.activeElement === last) {
        event.preventDefault();
        first?.focus();
      }
    }
    document.addEventListener("keydown", trapFocus);
    return () => {
      document.body.style.overflow = overflow;
      document.removeEventListener("keydown", trapFocus);
      previous?.focus();
    };
  }, [help]);

  const ask = useCallback(
    async (text: string) => {
      if (
        !text.trim() ||
        askBusy.current ||
        !camera.active ||
        !health ||
        health.ollama.status !== "ready"
      )
        return;
      let image: string;
      try {
        image = camera.capture(0.9);
      } catch (error) {
        setAskError(
          error instanceof Error
            ? error.message
            : "Could not capture a frame. Restart the camera.",
        );
        return;
      }
      askBusy.current = true;
      askStarted.current = performance.now();
      const controller = new AbortController();
      askController.current = controller;
      setAsking(true);
      setAskSeconds(0);
      setAskError("");
      setQuestion("");
      setMessages((previous) => [
        ...previous,
        {
          id: ++messageId.current,
          role: "user",
          text: text.trim(),
          time: timeNow(),
        },
      ]);
      try {
        const answer = await askFrame(
          image,
          text.trim(),
          language,
          controller.signal,
        );
        if (!controller.signal.aborted)
          setMessages((previous) => [
            ...previous,
            {
              id: ++messageId.current,
              role: "assistant",
              text: answer.answer,
              time: timeNow(),
              elapsed: answer.elapsed_ms,
              language,
            },
          ]);
      } catch (err) {
        if (!mounted.current) return;
        setQuestion((current) => current || text.trim());
        setMessages((previous) => [
          ...previous,
          {
            id: ++messageId.current,
            role: "assistant",
            text: controller.signal.aborted
              ? "Request cancelled. The local model may take a moment to release the request. Your question is ready to retry."
              : err instanceof Error
                ? err.message
                : "Something went wrong. Please try again.",
            time: timeNow(),
            error: true,
          },
        ]);
      } finally {
        askBusy.current = false;
        if (mounted.current) setAsking(false);
        askController.current = null;
      }
    },
    [camera.active, camera.capture, language, health],
  );

  const confidence = result.total
    ? Math.round(
        (result.objects.reduce((sum, object) => sum + object.confidence, 0) /
          result.total) *
          100,
      )
    : null;
  const connected = !!health;
  const visionReady = health?.ollama.status === "ready";
  const canAsk =
    camera.active && visionReady && !asking && !health?.ollama.busy;
  const detectorStatus = !health
    ? "Checking / offline"
    : health.detector.status === "ready"
      ? `Ready · ${health.detector.image_size}px · ${health.detector.device}`
      : health.detector.status === "loading"
        ? "Warming up…"
        : "Unavailable";
  const visionStatus = asking
    ? "Generating…"
    : !health
      ? "Checking / offline"
      : health.ollama.busy
        ? "Busy"
        : visionReady
          ? "Available locally"
          : health.ollama.status === "model_missing"
            ? "Model missing"
            : "Ollama offline";
  const boxesFresh =
    camera.active &&
    pageVisible &&
    now - lastDetectionAt < Math.max(4500, result.inference_ms * 2);
  const statusHint =
    backendError ||
    (health?.detector.status === "error" ? health.detector.error : "") ||
    (health?.detector.status === "loading"
      ? "YOLO is warming up. Your camera can start now; detection resumes automatically."
      : "") ||
    (health?.ollama.status === "offline"
      ? "Ollama is offline. Run ollama serve to enable AI chat."
      : health?.ollama.status === "model_missing"
        ? `Vision model missing. Run ollama pull ${health.ollama.model}`
        : "");
  const prompts =
    language === "en"
      ? [
          "What do you see?",
          "Describe this scene",
          "How many objects are here?",
        ]
      : [
          "Nimalarni ko‘ryapsiz?",
          "Bu manzarani tasvirlang",
          "Bu yerda nechta buyum bor?",
        ];

  function goToChat() {
    setView("chat");
    document
      .getElementById("assistant")
      ?.scrollIntoView({ behavior: "smooth", block: "center" });
    chatInput.current?.focus({ preventScroll: true });
  }

  return (
    <div className="app-shell">
      <a className="skip-link" href="#workspace">
        Skip to workspace
      </a>
      <aside className="navigation">
        <a className="wordmark" href="#workspace" aria-label="Vision workspace">
          <span aria-hidden="true">·</span> VISION
        </a>
        <nav aria-label="Main navigation">
          <button
            aria-label="Workspace"
            className={`nav-link ${view === "workspace" ? "is-selected" : ""}`}
            aria-current={view === "workspace" ? "page" : undefined}
            onClick={() => {
              setView("workspace");
              window.scrollTo({ top: 0, behavior: "smooth" });
            }}
          >
            <Grid2X2 size={18} strokeWidth={1.5} />
            <span>Workspace</span>
          </button>
          <button
            aria-label="Assistant"
            className={`nav-link ${view === "chat" ? "is-selected" : ""}`}
            aria-current={view === "chat" ? "page" : undefined}
            onClick={goToChat}
          >
            <MessageSquare size={18} strokeWidth={1.5} />
            <span>Assistant</span>
          </button>
        </nav>
        <div className="navigation-bottom">
          <button
            aria-label="Setup and models"
            className="nav-link"
            onClick={() => setHelp(true)}
          >
            <Settings2 size={18} strokeWidth={1.5} />
            <span>Setup & models</span>
          </button>
          <p>
            Private by default.
            <br />
            Nothing captured is saved.
          </p>
        </div>
      </aside>

      <main id="workspace" className="workspace" tabIndex={-1}>
        <header className="workspace-header">
          <div className="heading-copy">
            <p className="eyebrow">AI vision assistant</p>
            <h1>See. Understand. Interact.</h1>
            <p className="subtitle">
              Real-time object detection and visual understanding, powered by
              local AI.
            </p>
          </div>
          <div className="system-status" aria-label="Local service status">
            <span className="backend-status">
              <span
                className={`status-dot ${connected ? "is-ready" : healthChecked ? "is-offline" : ""}`}
              />
              {connected
                ? "Running locally"
                : healthChecked
                  ? "Backend offline"
                  : "Connecting"}
            </span>
            <span title={`YOLO11n: ${detectorStatus}`}>
              <span
                className={`status-dot ${health?.detector.status === "ready" ? "is-ready" : ""}`}
              />
              YOLO11n
            </span>
            <span title={`Qwen2.5-VL: ${visionStatus}`}>
              <span className={`status-dot ${visionReady ? "is-ready" : ""}`} />
              Qwen2.5-VL
            </span>
          </div>
        </header>

        {statusHint && (
          <div className="service-notice" role="status">
            <span>{statusHint}</span>
            <button onClick={() => setHelp(true)}>View setup</button>
          </div>
        )}

        <div className="workspace-grid">
          <section
            className="camera-column"
            aria-label="Live camera and detection statistics"
          >
            <div className="camera-panel" ref={cameraPanel}>
              <div
                className={`camera-viewport ${camera.active ? "is-live" : ""}`}
              >
                <CameraStage
                  width={camera.dimensions.width}
                  height={camera.dimensions.height}
                >
                  <video
                    ref={camera.videoRef}
                    autoPlay
                    muted
                    playsInline
                    className={camera.active ? "visible" : ""}
                    aria-label="Live webcam feed"
                  />
                  {camera.active && showBoxes && boxesFresh && (
                    <DetectionOverlay objects={result.objects} />
                  )}
                </CameraStage>
                <div className="camera-topline">
                  <span className="feed-state">
                    <span
                      className={`status-dot ${camera.active ? "is-ready" : ""}`}
                    />
                    {camera.active
                      ? "Live"
                      : camera.starting
                        ? "Connecting"
                        : "Camera off"}
                  </span>
                  <div className="camera-actions">
                    <button
                      className="icon-button camera-icon"
                      title="Fullscreen camera"
                      aria-label="Fullscreen camera"
                      onClick={() => {
                        if (document.fullscreenElement)
                          void document.exitFullscreen().catch(() => {});
                        else
                          void cameraPanel.current
                            ?.requestFullscreen?.()
                            .catch(() => {});
                      }}
                    >
                      <Maximize2 size={17} strokeWidth={1.6} />
                    </button>
                    <button
                      className="icon-button camera-icon"
                      title={
                        camera.active || camera.starting
                          ? "Stop camera"
                          : "Start camera"
                      }
                      aria-label={
                        camera.active || camera.starting
                          ? "Stop camera"
                          : "Start camera"
                      }
                      onClick={() =>
                        camera.active || camera.starting
                          ? camera.stop()
                          : void camera.start()
                      }
                    >
                      {camera.active || camera.starting ? (
                        <Square size={15} strokeWidth={1.5} />
                      ) : (
                        <Camera size={18} strokeWidth={1.5} />
                      )}
                    </button>
                  </div>
                </div>
                {!camera.active && (
                  <div className="camera-empty">
                    <span className="camera-empty-caption">
                      Your point of view
                    </span>
                    <h2>
                      A new way to see
                      <br />
                      what’s in front of you.
                    </h2>
                    <p>Start your camera. Let the conversation follow.</p>
                    <button
                      className="primary-button"
                      disabled={camera.starting}
                      onClick={() => void camera.start()}
                    >
                      {camera.starting ? (
                        <LoaderCircle className="spinner" size={16} />
                      ) : (
                        <Camera size={16} strokeWidth={1.6} />
                      )}
                      {camera.starting ? "Connecting camera…" : "Start camera"}
                    </button>
                    <small>Camera permission is always yours to control.</small>
                  </div>
                )}
                {camera.active && (
                  <div className="camera-bottomline">
                    <span>
                      {detectionError
                        ? "Detection unavailable"
                        : !pageVisible
                          ? "Detection paused"
                          : detecting
                            ? "Analyzing frame…"
                            : boxesFresh
                              ? "Latest analyzed frame"
                              : "Waiting for a fresh frame"}
                    </span>
                    <span>{camera.resolution}</span>
                  </div>
                )}
              </div>
              {(camera.error || detectionError) && (
                <p className="camera-error" role="alert">
                  {camera.error || detectionError}
                </p>
              )}
            </div>

            <dl className="metrics-bar" aria-label="Live detection statistics">
              <div>
                <dt>Objects</dt>
                <dd>{camera.active ? result.total : "—"}</dd>
              </div>
              <div>
                <dt>Classes</dt>
                <dd>
                  {camera.active ? Object.keys(result.counts).length : "—"}
                </dd>
              </div>
              <div>
                <dt>Avg. confidence</dt>
                <dd>{confidence === null ? "—" : `${confidence}%`}</dd>
              </div>
              <div
                title={
                  result.roundtrip_ms !== undefined
                    ? `${result.roundtrip_ms} ms browser round trip. The value shown measures YOLO prediction.`
                    : "YOLO prediction time, measured after warm-up"
                }
              >
                <dt>Detection latency</dt>
                <dd>
                  {result.inference_ms ? (
                    <>
                      {result.inference_ms}
                      <span>ms</span>
                    </>
                  ) : (
                    "—"
                  )}
                </dd>
              </div>
            </dl>
          </section>

          <section
            className="conversation-panel"
            id="assistant"
            aria-label="Vision assistant"
          >
            <header className="conversation-header">
              <div>
                <h2>
                  <span className="assistant-dot" />
                  Vision Assistant
                </h2>
                <p>Ask about what your camera sees.</p>
              </div>
              <button
                className="icon-button"
                title="Clear conversation"
                aria-label="Clear conversation"
                disabled={asking || !messages.length}
                onClick={() => setMessages([])}
              >
                <Trash2 size={16} strokeWidth={1.5} />
              </button>
            </header>
            <div
              className="conversation-scroll"
              role="log"
              aria-label="Conversation"
              aria-live="polite"
              aria-relevant="additions text"
            >
              {messages.length === 0 ? (
                <div className="conversation-empty">
                  <p className="empty-kicker">
                    A conversation with your surroundings
                  </p>
                  <h3>
                    What are you
                    <br />
                    curious about?
                  </h3>
                  <p>
                    Identify an object, explore a detail,
                    <br />
                    or make sense of the whole scene.
                  </p>
                  <span>
                    {camera.active
                      ? "Your camera is ready. Ask a question below."
                      : "Start your camera when you’re ready."}
                  </span>
                </div>
              ) : (
                messages.map((message) => (
                  <article
                    key={message.id}
                    className={`message ${message.role === "user" ? "message-user" : "message-assistant"} ${message.error ? "message-error" : ""}`}
                  >
                    <header className="message-meta">
                      <strong>
                        <span
                          className={
                            message.role === "user"
                              ? "user-dot"
                              : "assistant-dot"
                          }
                        />
                        {message.role === "user" ? "You" : "Vision"}
                      </strong>
                      <time>{message.time}</time>
                    </header>
                    <p>{message.text}</p>
                    {message.role === "user" && (
                      <span className="message-caption">
                        Camera frame attached
                      </span>
                    )}
                    {message.elapsed !== undefined && (
                      <span className="message-caption">
                        {(message.elapsed / 1000).toFixed(1)}s response
                        {message.language === "uz" &&
                          " · Uzbek, limited accuracy"}
                      </span>
                    )}
                  </article>
                ))
              )}
              {asking && (
                <div className="thinking" role="status">
                  <span className="thinking-indicator" />
                  <span>
                    {askSeconds >= 15
                      ? "Still analyzing. The first model load can take a moment."
                      : "Looking at your frame"}
                  </span>
                  <time>{askSeconds}s</time>
                </div>
              )}
              <div ref={chatEnd} />
            </div>
            <div className="conversation-compose">
              {messages.length === 0 && (
                <div className="suggestions" aria-label="Suggested questions">
                  {prompts.map((prompt) => (
                    <button
                      key={prompt}
                      disabled={!canAsk}
                      onClick={() => void ask(prompt)}
                    >
                      {prompt}
                    </button>
                  ))}
                </div>
              )}
              <form
                onSubmit={(event) => {
                  event.preventDefault();
                  void ask(question);
                }}
              >
                <div className="composer">
                  <textarea
                    ref={chatInput}
                    value={question}
                    onChange={(event) => setQuestion(event.target.value)}
                    maxLength={2000}
                    rows={2}
                    aria-label="Question about the camera"
                    placeholder={
                      language === "uz"
                        ? "Ko‘rayotgan narsalaringiz haqida so‘rang…"
                        : "Ask something about what you see…"
                    }
                    onKeyDown={(event) => {
                      if (
                        event.key === "Enter" &&
                        !event.shiftKey &&
                        !event.nativeEvent.isComposing
                      ) {
                        event.preventDefault();
                        void ask(question);
                      }
                    }}
                  />
                  <div className="composer-controls">
                    <span className="frame-context">
                      <span
                        className={`status-dot ${camera.active ? "is-ready" : ""}`}
                      />
                      {camera.active ? "Live frame on send" : "Camera is off"}
                    </span>
                    {asking ? (
                      <button
                        className="send-button cancel-button"
                        type="button"
                        onClick={() => askController.current?.abort()}
                        aria-label="Cancel request"
                        title="Cancel request"
                      >
                        <Square size={15} />
                      </button>
                    ) : (
                      <button
                        className="send-button"
                        type="submit"
                        disabled={!canAsk || !question.trim()}
                        aria-label="Send question"
                        title={
                          camera.active
                            ? "Send question · Enter"
                            : "Start the camera to ask a question"
                        }
                      >
                        <ArrowUp size={20} strokeWidth={1.7} />
                      </button>
                    )}
                  </div>
                </div>
              </form>
              {askError && (
                <p className="compose-notice is-error" role="alert">
                  {askError}
                </p>
              )}
              {language === "uz" && (
                <p className="compose-notice" role="status">
                  O‘zbek javoblari xato bo‘lishi mumkin. English is more
                  reliable with this 3B model.
                </p>
              )}
              {!visionReady && (
                <p className="compose-notice" role="status">
                  {visionStatus}. Chat is available when the local model is
                  ready.
                </p>
              )}
              <div className="compose-footer">
                <label>
                  <span className="sr-only">Answer language</span>
                  <select
                    aria-label="Answer language"
                    value={language}
                    onChange={(event) =>
                      setLanguage(event.target.value as "en" | "uz")
                    }
                  >
                    <option value="en">English</option>
                    <option value="uz">O‘zbek</option>
                  </select>
                </label>
                <span>Frames are never saved</span>
              </div>
            </div>
          </section>
        </div>

        <section className="detection-details" aria-label="Detected objects">
          <div className="detection-heading">
            <h2>In this frame</h2>
            <label className="box-switch">
              <input
                type="checkbox"
                checked={showBoxes}
                onChange={() => setShowBoxes((value) => !value)}
              />
              <span>Show boxes</span>
            </label>
            <span className="frame-count">{frames} frames analyzed</span>
          </div>
          {result.total ? (
            <ul className="object-list">
              {Object.entries(result.counts)
                .sort((a, b) => b[1] - a[1])
                .map(([label, count]) => {
                  const avg = Math.round(
                    (result.objects
                      .filter((object) => object.label === label)
                      .reduce((sum, object) => sum + object.confidence, 0) /
                      count) *
                      100,
                  );
                  return (
                    <li key={label}>
                      <span>{label}</span>
                      <span className="object-count">{count}</span>
                      <span className="object-confidence">{avg}%</span>
                    </li>
                  );
                })}
            </ul>
          ) : (
            <p className="detection-empty">
              {camera.active
                ? "No COCO objects detected in the latest frame. You can still ask about what you see."
                : "Detected objects will appear here when your camera is on."}
            </p>
          )}
        </section>
        <footer className="workspace-footer">
          <span>On your device. In your control.</span>
          <span>
            Detection samples every 1.5s, or when the previous frame finishes.
          </span>
        </footer>
      </main>

      {help && (
        <div
          className="modal-backdrop"
          onClick={(event) => {
            if (event.target === event.currentTarget) setHelp(false);
          }}
        >
          <section
            className="help-modal"
            role="dialog"
            aria-modal="true"
            aria-labelledby="help-title"
            onKeyDown={(event) => {
              if (event.key === "Escape") setHelp(false);
            }}
          >
            <header>
              <div>
                <p className="eyebrow">Your local workspace</p>
                <h2 id="help-title">Setup & models</h2>
              </div>
              <button
                autoFocus
                className="icon-button"
                aria-label="Close setup"
                onClick={() => setHelp(false)}
              >
                <X size={20} />
              </button>
            </header>
            <dl className="model-status-list">
              <div>
                <dt>Backend</dt>
                <dd>
                  <span
                    className={`status-dot ${connected ? "is-ready" : "is-offline"}`}
                  />
                  {connected
                    ? "Connected"
                    : healthChecked
                      ? "Offline"
                      : "Connecting"}
                </dd>
              </div>
              <div>
                <dt>YOLO11n</dt>
                <dd>{detectorStatus}</dd>
              </div>
              <div>
                <dt>Qwen2.5-VL · 3B</dt>
                <dd>{visionStatus}</dd>
              </div>
            </dl>
            <ol className="setup-steps">
              <li>
                <h3>Start the local services</h3>
                <p>FastAPI uses port 8000; Ollama uses port 11434.</p>
                <code>ollama pull qwen2.5vl:3b</code>
                <code>ollama serve</code>
                <small>Skip serve if Ollama is already running.</small>
              </li>
              <li>
                <h3>Connect your camera</h3>
                <p>
                  Start the camera and allow access in your browser. Use
                  localhost or HTTPS. Stop releases your camera.
                </p>
              </li>
              <li>
                <h3>Ask about your view</h3>
                <p>
                  A fresh frame accompanies each question. Choose English or
                  O‘zbek below the input. The first response can take longer
                  while the model loads.
                </p>
              </li>
            </ol>
            <p className="help-note">
              Frames stay in memory and are never saved. AI may misidentify
              objects. English is more reliable than Uzbek with this 3B model.
            </p>
            <button className="primary-button" onClick={() => setHelp(false)}>
              Back to workspace
            </button>
          </section>
        </div>
      )}
    </div>
  );
}
