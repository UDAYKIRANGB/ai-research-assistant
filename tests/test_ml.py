"""Tests for the ML pipeline: dataset loading/splitting (fast) and an
optional end-to-end training smoke test (slow, TensorFlow-heavy).

Run the slow training smoke test explicitly with:
    RUN_SLOW_TESTS=1 pytest tests/test_ml.py -m slow
"""
import os

import pytest

from src.ml.dataset_prep import get_category_names, load_dataset, prepare_train_test_split
from src.ml.predictor import DocumentClassifier

RUN_SLOW = os.environ.get("RUN_SLOW_TESTS") == "1"


class TestDatasetPrep:
    def test_load_dataset_returns_expected_columns(self):
        df = load_dataset()
        assert {"text", "category"}.issubset(df.columns)
        assert len(df) > 0

    def test_dataset_covers_all_seven_categories(self):
        df = load_dataset()
        categories = set(df["category"].unique())
        expected = {
            "Artificial Intelligence",
            "Machine Learning",
            "Computer Vision",
            "Natural Language Processing",
            "Robotics",
            "Cyber Security",
            "Cloud Computing",
        }
        assert expected.issubset(categories)

    def test_train_test_split_is_stratified_and_labels_encoded(self):
        df = load_dataset()
        x_train, x_test, y_train, y_test, encoder = prepare_train_test_split(df, test_size=0.2)

        assert len(x_train) == len(y_train)
        assert len(x_test) == len(y_test)
        assert len(x_train) + len(x_test) == len(df)

        category_names = get_category_names(encoder)
        assert len(category_names) == df["category"].nunique()


class TestDocumentClassifierPredictor:
    def test_predictor_reports_not_ready_when_no_model_file(self, tmp_path, monkeypatch):
        # Point settings at a location with no trained model yet.
        from config.settings import settings
        monkeypatch.setattr(settings, "tf_model_path", str(tmp_path / "missing.h5"))
        monkeypatch.setattr(settings, "tokenizer_path", str(tmp_path / "missing.pickle"))

        classifier = DocumentClassifier()
        assert classifier.is_ready() is False
        assert classifier.predict("some text") is None


@pytest.mark.slow
@pytest.mark.skipif(not RUN_SLOW, reason="Set RUN_SLOW_TESTS=1 to run the full TensorFlow training smoke test.")
def test_end_to_end_training_smoke_test(tmp_path, monkeypatch):
    """Trains a tiny model for 1 epoch to confirm the pipeline is wired
    correctly end-to-end (build -> train -> evaluate -> persist)."""
    from config.settings import settings
    monkeypatch.setattr(settings, "tf_model_path", str(tmp_path / "model.h5"))
    monkeypatch.setattr(settings, "tokenizer_path", str(tmp_path / "tokenizer.pickle"))

    from src.ml.train_classifier import build_and_train_classifier

    model, encoder, history = build_and_train_classifier(epochs=1, batch_size=8)

    assert model is not None
    assert os.path.exists(settings.tf_model_path)
    assert os.path.exists(settings.tokenizer_path)
    assert "accuracy" in history.history
