"""One shared model; serialize inference because YOLO predictors are stateful."""
from collections import Counter
from threading import Lock
from time import perf_counter

import numpy as np

from .config import YOLO_CONFIDENCE, YOLO_DEVICE, YOLO_IMAGE_SIZE, YOLO_MODEL


class Detector:
    def __init__(self):
        self.model = None
        self.error: str | None = None
        self.lock = Lock()

    def load(self):
        with self.lock:
            if self.model is None:
                try:
                    from ultralytics import YOLO
                    model = YOLO(YOLO_MODEL)
                    # Absorb first-inference setup before reporting readiness.
                    model.predict(
                        np.zeros((YOLO_IMAGE_SIZE, YOLO_IMAGE_SIZE, 3), dtype=np.uint8),
                        **self.predict_options(),
                    )
                    self.model = model
                    self.error = None
                except Exception:
                    self.error = "YOLO could not load. Check the model path and first-run download."
                    raise

    def detect(self, frame: np.ndarray) -> dict:
        if self.model is None:
            self.load()
        with self.lock:
            start = perf_counter()
            result = self.model.predict(frame, **self.predict_options())[0]
            inference_ms = round((perf_counter() - start) * 1000, 1)
            height, width = frame.shape[:2]
            objects = []
            # One transfer per tensor rather than one GPU synchronization per box.
            boxes = result.boxes
            coordinates = boxes.xyxy.cpu().numpy() / np.array([width, height, width, height])
            coordinates = np.clip(coordinates, 0, 1)
            for bounds, confidence, class_id in zip(coordinates, boxes.conf.cpu().tolist(), boxes.cls.cpu().tolist()):
                objects.append({
                    "label": result.names[int(class_id)],
                    "confidence": round(confidence, 4),
                    "box": bounds.tolist(),
                })
        return {
            "objects": objects,
            "counts": dict(Counter(obj["label"] for obj in objects)),
            "total": len(objects),
            "inference_ms": inference_ms,
            "width": width,
            "height": height,
        }

    @staticmethod
    def predict_options() -> dict:
        # Single-scale, rectangular padding; leave GPU memory available to Ollama.
        return {
            "conf": YOLO_CONFIDENCE, "imgsz": YOLO_IMAGE_SIZE, "device": YOLO_DEVICE,
            "rect": True, "max_det": 100, "augment": False, "verbose": False,
            "save": False, "save_txt": False, "save_crop": False,
        }


detector = Detector()

