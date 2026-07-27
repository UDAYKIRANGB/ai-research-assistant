"""Dataset preparation for the TensorFlow document-classification model.

Expects a CSV at `data/dataset/training_data.csv` with two columns:
    text, category

A small sample dataset (data/dataset/training_data.csv) covering the seven
target categories is included so `train_classifier.py` runs out of the box.
For production use, replace it with a larger labelled corpus (e.g. arXiv
abstracts) - the loader and pipeline do not need to change.
"""
import os
from typing import List, Tuple

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

from config.settings import settings
from src.logging_config import get_logger

logger = get_logger(__name__)

DEFAULT_DATASET_PATH = os.path.join(settings.dataset_dir, "training_data.csv")


def load_dataset(csv_path: str = DEFAULT_DATASET_PATH) -> pd.DataFrame:
    if not os.path.exists(csv_path):
        raise FileNotFoundError(
            f"Training dataset not found at {csv_path}. "
            "Provide a CSV with 'text' and 'category' columns."
        )
    df = pd.read_csv(csv_path).dropna(subset=["text", "category"])
    logger.info("Loaded dataset with %d rows and %d categories", len(df), df["category"].nunique())
    return df


def prepare_train_test_split(
    df: pd.DataFrame, test_size: float = 0.2, random_state: int = 42
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, LabelEncoder]:
    encoder = LabelEncoder()
    labels = encoder.fit_transform(df["category"].values)
    texts = df["text"].astype(str).values

    x_train, x_test, y_train, y_test = train_test_split(
        texts, labels, test_size=test_size, random_state=random_state, stratify=labels
    )
    return x_train, x_test, y_train, y_test, encoder


def get_category_names(encoder: LabelEncoder) -> List[str]:
    return list(encoder.classes_)
