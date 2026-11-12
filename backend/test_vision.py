"""Regression tests for upstream repeat-aborts, diagnostics and bounded recovery."""

import asyncio
import json
import unittest
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import httpx
import numpy as np
from fastapi import HTTPException

from backend.main import AskRequest, ask
from backend.vision import ask_vision


class VisionRecoveryTests(unittest.IsolatedAsyncioTestCase):
    async def run_request(self, handler):
        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
            return await ask_vision(
                client, "c2VjcmV0LWltYWdl", "private-question", "en"
            )

    async def test_repeat_abort_resets_same_model_then_retries_once(self):
        calls = []

        def handle(request):
            calls.append((request.url.path, json.loads(request.content)))
            if len(calls) == 1:
                return httpx.Response(
                    500,
                    json={"error": "prediction aborted, token repeat limit reached"},
                )
            if request.url.path == "/api/generate":
                return httpx.Response(200, json={"done": True, "done_reason": "unload"})
            return httpx.Response(
                200, json={"message": {"content": "A bus."}, "done": True}
            )

        with self.assertLogs("uvicorn.error.vision", level="INFO") as logs:
            self.assertEqual(await self.run_request(handle), "A bus.")
        self.assertEqual(
            [path for path, _ in calls], ["/api/chat", "/api/generate", "/api/chat"]
        )
        self.assertEqual(calls[0][1], calls[2][1])
        self.assertEqual(calls[0][1]["model"], calls[1][1]["model"])
        self.assertEqual(calls[1][1]["keep_alive"], 0)
        text = "\n".join(logs.output)
        for expected in [
            "status=500",
            "token repeat limit reached",
            "duration_ms=",
            "timeout_s=",
            "outcome=ok",
        ]:
            self.assertIn(expected, text)
        self.assertNotIn("private-question", text)
        self.assertNotIn("c2VjcmV0LWltYWdl", text)

    async def test_second_repeat_abort_is_reported_without_more_retries(self):
        calls = []

        def handle(request):
            calls.append(request.url.path)
            return (
                httpx.Response(200, json={"done": True})
                if request.url.path == "/api/generate"
                else httpx.Response(
                    500,
                    json={"error": "prediction aborted, token repeat limit reached"},
                )
            )

        with self.assertRaises(HTTPException) as error:
            await self.run_request(handle)
        self.assertEqual(error.exception.status_code, 502)
        self.assertIn("repetition limit", error.exception.detail)
        self.assertEqual(calls, ["/api/chat", "/api/generate", "/api/chat"])

    async def test_permanent_or_unclassified_errors_are_not_retried(self):
        for status, message, api_status, detail in [
            (
                500,
                "private-question c2VjcmV0LWltYWdl key=private-value",
                502,
                "HTTP 500",
            ),
            (500, "CUDA out of memory", 503, "insufficient memory"),
            (400, "failed to decode image", 502, "JPEG image"),
            (404, "model not found", 503, "model missing"),
            (429, "too many requests", 503, "HTTP 429"),
        ]:
            with self.subTest(status=status, message=message):
                calls = []

                def handle(request):
                    calls.append(request)
                    return httpx.Response(status, json={"error": message})

                with self.assertLogs(
                    "uvicorn.error.vision", level="INFO"
                ) as logs, self.assertRaises(HTTPException) as error:
                    await self.run_request(handle)
                self.assertEqual(error.exception.status_code, api_status)
                self.assertIn(detail, error.exception.detail)
                self.assertEqual(len(calls), 1)
                text = "\n".join(logs.output)
                self.assertNotIn("private-question", text)
                self.assertNotIn("private-value", text)
                self.assertNotIn("c2VjcmV0LWltYWdl", text)

    async def test_transport_interruption_retries_once(self):
        calls = []

        def handle(request):
            calls.append(request)
            if len(calls) == 1:
                raise httpx.RemoteProtocolError("sensitive transport text")
            return httpx.Response(200, json={"message": {"content": "A pen."}})

        self.assertEqual(await self.run_request(handle), "A pen.")
        self.assertEqual(len(calls), 2)

    async def test_timeout_is_logged_and_not_retried(self):
        calls = []

        def handle(request):
            calls.append(request)
            raise httpx.ReadTimeout("sensitive transport text")

        with self.assertLogs(
            "uvicorn.error.vision", level="INFO"
        ) as logs, self.assertRaises(HTTPException) as error:
            await self.run_request(handle)
        self.assertEqual(error.exception.status_code, 504)
        self.assertIn("ReadTimeout", error.exception.detail)
        self.assertEqual(len(calls), 1)
        self.assertIn("ReadTimeout", "\n".join(logs.output))
        self.assertNotIn("sensitive transport text", "\n".join(logs.output))

    async def test_total_deadline_includes_recovery(self):
        calls = []

        async def handle(request):
            calls.append(request.url.path)
            if request.url.path == "/api/chat":
                return httpx.Response(
                    500,
                    json={"error": "prediction aborted, token repeat limit reached"},
                )
            await asyncio.sleep(10)
            return httpx.Response(200, json={"done": True})

        with patch("backend.vision.OLLAMA_TIMEOUT", 0.03), self.assertRaises(
            HTTPException
        ) as error:
            await self.run_request(handle)
        self.assertEqual(error.exception.status_code, 504)
        self.assertEqual(calls, ["/api/chat", "/api/generate"])

    async def test_failed_reset_does_not_retry_inference(self):
        calls = []

        def handle(request):
            calls.append(request.url.path)
            return httpx.Response(
                500, json={"error": "prediction aborted, token repeat limit reached"}
            )

        with self.assertRaises(HTTPException) as error:
            await self.run_request(handle)
        self.assertEqual(error.exception.status_code, 503)
        self.assertEqual(calls, ["/api/chat", "/api/generate"])

    async def test_busy_503_retries_once(self):
        calls = []

        def handle(request):
            calls.append(request)
            if len(calls) == 1:
                return httpx.Response(503, json={"error": "server busy"})
            return httpx.Response(200, json={"message": {"content": "A watch."}})

        self.assertEqual(await self.run_request(handle), "A watch.")
        self.assertEqual(len(calls), 2)

    async def test_app_gate_rejects_overlap_while_inference_is_pending(self):
        entered, release = asyncio.Event(), asyncio.Event()

        async def slow_vision(*_):
            entered.set()
            await release.wait()
            return "A bus."

        request = SimpleNamespace(
            app=SimpleNamespace(state=SimpleNamespace(client=None)),
            is_disconnected=AsyncMock(return_value=False),
        )
        with patch("backend.main.ask_vision", side_effect=slow_vision), patch(
            "backend.main.decode_image", return_value=np.zeros((32, 32, 3), np.uint8)
        ):
            first = asyncio.create_task(
                ask(AskRequest(image="frame", question="Describe."), request)
            )
            try:
                await asyncio.wait_for(entered.wait(), 2)
                with self.assertRaises(HTTPException) as error:
                    await ask(
                        AskRequest(image="frame", question="Second question."), request
                    )
                self.assertEqual(error.exception.status_code, 429)
            finally:
                release.set()
                result = await first
        self.assertEqual(result["answer"], "A bus.")


if __name__ == "__main__":
    unittest.main()
