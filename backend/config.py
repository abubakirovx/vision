import os
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent.parent
load_dotenv(ROOT / ".env")

OLLAMA_URL = os.getenv("OLLAMA_URL", "http://127.0.0.1:11434").rstrip("/")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "qwen2.5vl:3b")
OLLAMA_TIMEOUT = float(os.getenv("OLLAMA_TIMEOUT", "180"))
OLLAMA_NUM_CTX = int(os.getenv("OLLAMA_NUM_CTX", "4096"))
OLLAMA_MAX_TOKENS = int(os.getenv("OLLAMA_MAX_TOKENS", "384"))
YOLO_MODEL = os.getenv("YOLO_MODEL", "yolo11n.pt")
YOLO_CONFIDENCE = float(os.getenv("YOLO_CONFIDENCE", "0.30"))
YOLO_IMAGE_SIZE = int(os.getenv("YOLO_IMAGE_SIZE", "768"))
YOLO_DEVICE = os.getenv("YOLO_DEVICE", "cpu")
if not 0 < YOLO_CONFIDENCE <= 1:
    raise ValueError("YOLO_CONFIDENCE must be between 0 and 1.")
if YOLO_IMAGE_SIZE not in (640, 768, 960):
    raise ValueError("YOLO_IMAGE_SIZE must be 640, 768, or 960.")
if OLLAMA_TIMEOUT <= 0 or not 2048 <= OLLAMA_NUM_CTX <= 8192 or not 64 <= OLLAMA_MAX_TOKENS <= 1024:
    raise ValueError("Check OLLAMA_TIMEOUT, OLLAMA_NUM_CTX (2048–8192), and OLLAMA_MAX_TOKENS (64–1024).")
CORS_ORIGINS = os.getenv(
    "CORS_ORIGINS", "http://localhost:5173,http://127.0.0.1:5173"
).split(",")
CORS_ORIGINS = [origin.strip() for origin in CORS_ORIGINS if origin.strip()]
MAX_IMAGE_BYTES = 5 * 1024 * 1024
MAX_REQUEST_BYTES = 7 * 1024 * 1024 + 8192

