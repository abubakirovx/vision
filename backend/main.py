import asyncio
import logging
from contextlib import asynccontextmanager
from time import perf_counter
from typing import Literal

import httpx
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from starlette.concurrency import run_in_threadpool
from starlette.responses import JSONResponse

from .config import CORS_ORIGINS, MAX_REQUEST_BYTES, OLLAMA_MODEL, YOLO_CONFIDENCE, YOLO_DEVICE, YOLO_IMAGE_SIZE
from .detector import detector
from .images import decode_image, encode_jpeg
from .vision import ask_vision, ollama_status

logger = logging.getLogger("vision")


@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.client = httpx.AsyncClient(
        trust_env=False, limits=httpx.Limits(max_connections=4, max_keepalive_connections=4)
    )
    # Warm up in the background; /health remains available during model download.
    async def load_detector():
        try:
            await run_in_threadpool(detector.load)
        except Exception:
            logger.warning("YOLO unavailable; detection requests will retry model loading.")
    loader = asyncio.create_task(load_detector())
    yield
    loader.cancel()
    await asyncio.gather(loader, return_exceptions=True)
    await app.state.client.aclose()


app = FastAPI(title="Vision Assistant", version="1.0.0", lifespan=lifespan)
detect_gate = asyncio.Semaphore(1)
ask_gate = asyncio.Semaphore(1)


class RequestSizeLimit:
    """Bound JSON bodies before parsing, including chunked requests."""
    def __init__(self, application):
        self.app = application

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http" or scope["method"] != "POST":
            return await self.app(scope, receive, send)
        body = bytearray()
        while True:
            message = await receive()
            if message["type"] == "http.disconnect":
                return
            body.extend(message.get("body", b""))
            if len(body) > MAX_REQUEST_BYTES:
                return await JSONResponse({"detail": "Request exceeds the 7 MB limit."}, status_code=413)(scope, receive, send)
            if not message.get("more_body", False):
                break
        sent = False
        async def replay():
            nonlocal sent
            if not sent:
                sent = True
                return {"type": "http.request", "body": bytes(body), "more_body": False}
            return await receive()
        await self.app(scope, replay, send)


app.add_middleware(RequestSizeLimit)
# Include CORS headers even on rejected oversized requests.
app.add_middleware(CORSMiddleware, allow_origins=CORS_ORIGINS, allow_methods=["GET", "POST"], allow_headers=["Content-Type"])


class ImageRequest(BaseModel):
    image: str = Field(min_length=1, max_length=7_000_000)


class AskRequest(ImageRequest):
    question: str = Field(min_length=1, max_length=2000)
    language: Literal["en", "uz"] = "en"


@app.get("/health")
async def health(request: Request):
    return {
        "status": "ok",
        "detector": {"status": "ready" if detector.model is not None else ("error" if detector.error else "loading"), "model": "YOLO11n", "error": detector.error,
                     "image_size": YOLO_IMAGE_SIZE, "confidence": YOLO_CONFIDENCE, "device": YOLO_DEVICE},
        "ollama": {**await ollama_status(request.app.state.client), "busy": ask_gate.locked()},
    }


@app.post("/api/detect")
async def detect(payload: ImageRequest):
    if detect_gate.locked():
        raise HTTPException(429, "Detection is busy. Try the next frame.")
    async with detect_gate:
        start = perf_counter()
        if detector.model is None and not detector.error:
            raise HTTPException(503, "YOLO is warming up. Detection will resume automatically.")
        frame = await run_in_threadpool(decode_image, payload.image)
        try:
            result = await run_in_threadpool(detector.detect, frame)
            return {**result, "processing_ms": round((perf_counter() - start) * 1000, 1)}
        except Exception:
            logger.exception("Detection failed")
            raise HTTPException(503, "YOLO is unavailable. Check the model download and backend console.") from None


@app.post("/api/ask")
async def ask(payload: AskRequest, request: Request):
    question = payload.question.strip()
    if not question:
        raise HTTPException(422, "Please enter a question.")
    if ask_gate.locked():
        raise HTTPException(429, "The vision model is busy. Wait for the current answer.")
    async with ask_gate:
        start = perf_counter()
        frame = await run_in_threadpool(decode_image, payload.image)
        image = await run_in_threadpool(encode_jpeg, frame)
        # Closing the browser or cancelling a question closes the upstream HTTP request too.
        inference = asyncio.create_task(ask_vision(request.app.state.client, image, question, payload.language))
        async def disconnected():
            while not await request.is_disconnected():
                await asyncio.sleep(0.25)
        disconnect = asyncio.create_task(disconnected())
        try:
            done, _ = await asyncio.wait({inference, disconnect}, return_when=asyncio.FIRST_COMPLETED)
            if inference not in done:
                raise HTTPException(499, "Client disconnected; vision request cancelled.")
            answer = await inference
        finally:
            for task in (inference, disconnect):
                task.cancel()
            await asyncio.gather(inference, disconnect, return_exceptions=True)
        return {"answer": answer, "language": payload.language, "model": OLLAMA_MODEL, "elapsed_ms": round((perf_counter() - start) * 1000)}

