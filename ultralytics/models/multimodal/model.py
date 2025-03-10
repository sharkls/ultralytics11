# Ultralytics YOLO 🚀, AGPL-3.0 license

from pathlib import Path

from ultralytics.engine.model import Model
from ultralytics.models import yolo
from ultralytics.nn.tasks import  MultiModalDetectionModel
from ultralytics.utils import ROOT, yaml_load

from .predict import MultiModalDetectionPredictor
from .train import MultiModalDetectionTrainer
from .val import MultiModalDetectionValidator

class YOLOMultimodal(Model):
    """YOLO (You Only Look Once) multimodal model."""

    def __init__(self, model="yolo11n.pt", task=None, verbose=False):
        """Initialize YOLO model"""
        path = Path(model)
        print("task: ", task)
        super().__init__(model=model, task=task, verbose=verbose)    # Continue with default YOLO initialization

    @property
    def task_map(self):
        """Map head to model, trainer, validator, and predictor classes."""
        return {
            "multimodal": {
                "model": MultiModalDetectionModel,
                "trainer": MultiModalDetectionTrainer,
                "validator": MultiModalDetectionValidator,
                "predictor": MultiModalDetectionPredictor,
            },
        }