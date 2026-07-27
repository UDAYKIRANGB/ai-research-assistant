"""Model loading and inference wrapper. Used by the document upload pipeline
to auto-classify newly uploaded PDFs into a predefined category."""
import os
import pickle
from functools import lru_cache
from typing import Optional, Tuple

import numpy as np

from config.settings import settings
from src.logging_config import get_logger

logger = get_logger(__name__)


class DocumentClassifier:
    def __init__(self):
        self.model = None
        self.encoder = None
        self._load()

    def _load(self) -> None:
        if not (os.path.exists(settings.tf_model_path) and os.path.exists(settings.tokenizer_path)):
            logger.warning(
                "No trained classifier found at %s. Run `python -m src.ml.train_classifier` "
                "first. Classification will be skipped until a model exists.",
                settings.tf_model_path,
            )
            return

        import tensorflow as tf  # local import: TF is heavy, load only when needed

        self.model = tf.keras.models.load_model(settings.tf_model_path)
        with open(settings.tokenizer_path, "rb") as f:
            self.encoder = pickle.load(f)
        logger.info("Loaded trained classifier from %s", settings.tf_model_path)

    def is_ready(self) -> bool:
        return self.model is not None and self.encoder is not None

    def predict(self, text: str) -> Optional[Tuple[str, float]]:
        """Returns (category_label, confidence) or None if no model is loaded."""
        if not self.is_ready():
            return None
        probs = self.model.predict(np.array([text]), verbose=0)[0]
        idx = int(np.argmax(probs))
        label = self.encoder.inverse_transform([idx])[0]
        confidence = float(probs[idx])
        return label, confidence


@lru_cache
def get_classifier() -> DocumentClassifier:
    return DocumentClassifier()
