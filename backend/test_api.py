"""Contract/error tests use synthetic in-memory images and mocked model services."""
import base64
import asyncio
import json
from types import SimpleNamespace
import unittest
from unittest.mock import AsyncMock, patch

import cv2
import httpx
import numpy as np
from fastapi.testclient import TestClient

from backend.config import OLLAMA_MODEL
from backend.detector import Detector, detector
from backend.images import decode_image
from backend.main import AskRequest, app, ask, ask_gate, detect_gate
from backend.vision import ask_vision, ollama_status


class ApiTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        _, image = cv2.imencode('.jpg', np.zeros((64, 96, 3), dtype=np.uint8))
        cls.image = 'data:image/jpeg;base64,' + base64.b64encode(image).decode()

    def setUp(self):
        self.model = patch.object(detector, 'model', object())
        self.model.start()
        self.load = patch.object(detector, 'load')
        self.load.start()
        self.client = TestClient(app)
        self.client.__enter__()

    def tearDown(self):
        self.client.__exit__(None, None, None)
        self.load.stop()
        self.model.stop()

    def test_health_reports_dependencies(self):
        with patch('backend.main.ollama_status', new=AsyncMock(return_value={'status': 'offline', 'model': OLLAMA_MODEL})):
            response = self.client.get('/health')
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()['ollama']['status'], 'offline')

    def test_detect_contract_and_memory_decode(self):
        expected = {'objects': [{'label': 'cup', 'confidence': .9, 'box': [.1, .2, .5, .8]}], 'counts': {'cup': 1}, 'total': 1, 'inference_ms': 20, 'width': 96, 'height': 64}
        with patch.object(detector, 'detect', return_value=expected) as detect:
            response = self.client.post('/api/detect', json={'image': self.image})
            self.assertEqual(detect.call_args.args[0].shape, (64, 96, 3))
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertGreaterEqual(data.pop('processing_ms'), 0)
        self.assertEqual(data, expected)

    def test_corrupt_images_rejected(self):
        for image in ['not-base64', 'data:image/svg+xml;base64,PHN2Zz4=', 'dGVzdA==']:
            with self.subTest(image=image):
                self.assertEqual(self.client.post('/api/detect', json={'image': image}).status_code, 422)

    def test_detection_failure_is_actionable(self):
        with patch.object(detector, 'detect', side_effect=RuntimeError('unavailable')):
            response = self.client.post('/api/detect', json={'image': self.image})
        self.assertEqual(response.status_code, 503)

    def test_ask_forwards_fresh_image_and_uzbek(self):
        with patch('backend.main.ask_vision', new=AsyncMock(return_value='Bu qora rasm.')) as ask:
            response = self.client.post('/api/ask', json={'image': self.image, 'question': '  Nima bu?  ', 'language': 'uz'})
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()['answer'], 'Bu qora rasm.')
        self.assertEqual(ask.call_args.args[2:], ('Nima bu?', 'uz'))
        self.assertTrue(ask.call_args.args[1])

    def test_invalid_questions_and_languages(self):
        for fields in [{'question': ' '}, {'question': 'hello', 'language': 'de'}, {'question': 'x' * 2001}]:
            self.assertEqual(self.client.post('/api/ask', json={'image': self.image, **fields}).status_code, 422)

    def test_oversized_request_rejected_before_parsing(self):
        response = self.client.post('/api/detect', content=b'x' * (7 * 1024 * 1024 + 8193))
        self.assertEqual(response.status_code, 413)

    def test_busy_requests_are_rejected_without_inference(self):
        for path, gate in [('/api/detect', 'detect_gate'), ('/api/ask', 'ask_gate')]:
            with self.subTest(path=path), patch(f'backend.main.{gate}', asyncio.Semaphore(0)):
                response = self.client.post(path, json={'image': self.image, 'question': 'What is here?'})
                self.assertEqual(response.status_code, 429)

    def test_detector_loading_returns_recoverable_error(self):
        with patch.object(detector, 'model', None), patch.object(detector, 'error', None):
            response = self.client.post('/api/detect', json={'image': self.image})
        self.assertEqual(response.status_code, 503)
        self.assertIn('warming up', response.json()['detail'])

    def test_large_image_dimensions_and_resize(self):
        for size, status in [((8, 32), 422), ((32, 4097), 422)]:
            _, image = cv2.imencode('.png', np.zeros((*size, 3), np.uint8))
            response = self.client.post('/api/detect', json={'image': base64.b64encode(image).decode()})
            self.assertEqual(response.status_code, status)
        _, image = cv2.imencode('.jpg', np.zeros((1600, 800, 3), np.uint8))
        self.assertEqual(decode_image(base64.b64encode(image).decode()).shape, (1280, 640, 3))

    def test_oversize_response_keeps_cors_headers(self):
        response = self.client.post('/api/detect', content=b'x' * (7 * 1024 * 1024 + 8193), headers={'Origin': 'http://localhost:5173'})
        self.assertEqual(response.status_code, 413)
        self.assertEqual(response.headers['access-control-allow-origin'], 'http://localhost:5173')


class OllamaTests(unittest.IsolatedAsyncioTestCase):
    async def test_image_and_language_reach_ollama(self):
        async def handle(request):
            body = json.loads(request.content)
            self.assertFalse(body['stream'])
            self.assertEqual(body['messages'][1]['images'], ['image-bytes'])
            self.assertIn('Uzbek', body['messages'][0]['content'])
            self.assertIn('watches and pens', body['messages'][0]['content'])
            self.assertNotIn('Useful vocabulary', body['messages'][0]['content'])
            return httpx.Response(200, json={'message': {'content': 'Salom'}})
        async with httpx.AsyncClient(transport=httpx.MockTransport(handle)) as client:
            self.assertEqual(await ask_vision(client, 'image-bytes', 'Nima bu?', 'uz'), 'Salom')

    async def test_missing_model_returns_recovery(self):
        from fastapi import HTTPException
        async with httpx.AsyncClient(transport=httpx.MockTransport(lambda _: httpx.Response(404))) as client:
            with self.assertRaises(HTTPException) as error:
                await ask_vision(client, 'image', 'What is this?', 'en')
            self.assertEqual(error.exception.status_code, 503)
            self.assertIn('ollama pull', error.exception.detail)

    async def test_offline_and_timeout(self):
        from fastapi import HTTPException
        for exception, status in [(httpx.ConnectError('offline'), 503), (httpx.ReadTimeout('slow'), 504)]:
            def fail(_):
                raise exception
            async with httpx.AsyncClient(transport=httpx.MockTransport(fail)) as client:
                with self.assertRaises(HTTPException) as error:
                    await ask_vision(client, 'image', 'Question', 'en')
                self.assertEqual(error.exception.status_code, status)

    async def test_model_readiness(self):
        for models, status in [([], 'model_missing'), ([{'name': OLLAMA_MODEL}], 'ready')]:
            async with httpx.AsyncClient(transport=httpx.MockTransport(lambda _: httpx.Response(200, json={'models': models}))) as client:
                self.assertEqual((await ollama_status(client))['status'], status)

    async def test_english_and_empty_or_malformed_answers(self):
        from fastapi import HTTPException
        async def handle(request):
            body = json.loads(request.content)
            self.assertIn('English', body['messages'][0]['content'])
            self.assertIn('Respond only in English', body['messages'][1]['content'])
            return httpx.Response(200, json={'message': {'content': 'A pen.'}})
        async with httpx.AsyncClient(transport=httpx.MockTransport(handle)) as client:
            self.assertEqual(await ask_vision(client, 'image', 'What is this?', 'en'), 'A pen.')
        for body in [{'message': {'content': ''}}, {'message': {'content': None}}, []]:
            async with httpx.AsyncClient(transport=httpx.MockTransport(lambda _: httpx.Response(200, json=body))) as client:
                with self.assertRaises(HTTPException) as error:
                    await ask_vision(client, 'image', 'Question', 'en')
                self.assertEqual(error.exception.status_code, 502)

    async def test_disconnect_cancels_upstream_and_releases_gate(self):
        from fastapi import HTTPException
        cancelled = asyncio.Event()
        async def slow_vision(*_):
            try:
                await asyncio.sleep(30)
            finally:
                cancelled.set()
        request = SimpleNamespace(app=SimpleNamespace(state=SimpleNamespace(client=None)), is_disconnected=AsyncMock(return_value=True))
        with patch('backend.main.ask_vision', side_effect=slow_vision), patch('backend.main.decode_image', return_value=np.zeros((32, 32, 3), np.uint8)):
            with self.assertRaises(HTTPException) as error:
                await ask(AskRequest(image='in-memory', question='What is visible?'), request)
        self.assertEqual(error.exception.status_code, 499)
        self.assertTrue(cancelled.is_set())
        self.assertFalse(ask_gate.locked())
        self.assertFalse(detect_gate.locked())


class DetectorTests(unittest.TestCase):
    def test_normalized_boxes_match_original_frame_and_are_clamped(self):
        class Tensor:
            def __init__(self, values): self.values = np.array(values)
            def cpu(self): return self
            def numpy(self): return self.values
            def tolist(self): return self.values.tolist()
        boxes = SimpleNamespace(xyxy=Tensor([[-10, 20, 110, 180]]), conf=Tensor([0.92346]), cls=Tensor([0]))
        model = SimpleNamespace(predict=unittest.mock.Mock(return_value=[SimpleNamespace(boxes=boxes, names={0: 'cup'})]))
        local = Detector()
        local.model = model
        result = local.detect(np.zeros((200, 100, 3), np.uint8))
        self.assertEqual(result['objects'][0]['box'], [0, .1, 1, .9])
        self.assertEqual(result['counts'], {'cup': 1})
        self.assertEqual(result['objects'][0]['confidence'], .9235)
        self.assertFalse(model.predict.call_args.kwargs['save'])
        self.assertFalse(model.predict.call_args.kwargs['augment'])

    def test_empty_frame_result_has_no_boxes(self):
        class EmptyTensor:
            def cpu(self): return self
            def numpy(self): return np.empty((0, 4))
            def tolist(self): return []
        boxes = SimpleNamespace(xyxy=EmptyTensor(), conf=EmptyTensor(), cls=EmptyTensor())
        local = Detector()
        local.model = SimpleNamespace(predict=lambda *_, **__: [SimpleNamespace(boxes=boxes, names={})])
        result = local.detect(np.zeros((200, 100, 3), np.uint8))
        self.assertEqual(result['objects'], [])
        self.assertEqual(result['total'], 0)


if __name__ == '__main__':
    unittest.main()
