"""
TensorFlow document classifier: training, evaluation, and persistence.

Run directly as a script:
    python -m src.ml.train_classifier

Pipeline stages (per assignment requirement 4.7):
    1. Data preprocessing   -> src/ml/dataset_prep.py
    2. Feature engineering  -> Keras TextVectorization layer (built into the model)
    3. Model training       -> build_and_train_classifier()
    4. Model evaluation     -> evaluate_model()
    5. Model persistence    -> model.save(TF_MODEL_PATH) + label encoder pickle
    6. Prediction API       -> src/ml/predictor.py
"""
import os
import pickle

import numpy as np
import tensorflow as tf
from sklearn.metrics import classification_report
from tensorflow.keras import layers, models

from config.settings import settings
from src.logging_config import get_logger
from src.ml.dataset_prep import get_category_names, load_dataset, prepare_train_test_split

logger = get_logger(__name__)


def build_model(train_texts, num_classes: int, vocab_size: int = 10000, max_len: int = 200) -> tf.keras.Model:
    """Builds an end-to-end text classifier: raw text in, class probabilities out."""
    vectorize_layer = layers.TextVectorization(
        max_tokens=vocab_size,
        output_mode="int",
        output_sequence_length=max_len,
    )
    vectorize_layer.adapt(train_texts)

    model = models.Sequential(
        [
            layers.Input(shape=(1,), dtype=tf.string),
            vectorize_layer,
            layers.Embedding(vocab_size, 64, mask_zero=True),
            layers.GlobalAveragePooling1D(),
            layers.Dense(128, activation="relu"),
            layers.Dropout(0.3),
            layers.Dense(64, activation="relu"),
            layers.Dropout(0.3),
            layers.Dense(num_classes, activation="softmax"),
        ]
    )
    model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    return model


def build_and_train_classifier(epochs: int = 15, batch_size: int = 16):
    df = load_dataset()
    x_train, x_test, y_train, y_test, encoder = prepare_train_test_split(df)
    category_names = get_category_names(encoder)
    num_classes = len(category_names)

    logger.info("Training classifier on %d samples across %d categories: %s",
                len(x_train), num_classes, category_names)

    model = build_model(x_train, num_classes=num_classes)

    history = model.fit(
        x_train,
        y_train,
        epochs=epochs,
        batch_size=batch_size,
        validation_split=0.15,
        verbose=2,
    )

    evaluate_model(model, x_test, y_test, category_names)
    persist_model(model, encoder)
    return model, encoder, history


def evaluate_model(model, x_test, y_test, category_names) -> None:
    loss, accuracy = model.evaluate(x_test, y_test, verbose=0)
    logger.info("Test loss=%.4f accuracy=%.4f", loss, accuracy)

    y_pred_probs = model.predict(x_test, verbose=0)
    y_pred = np.argmax(y_pred_probs, axis=1)
    report = classification_report(y_test, y_pred, target_names=category_names, zero_division=0)
    logger.info("Classification report:\n%s", report)


def persist_model(model, encoder) -> None:
    os.makedirs(os.path.dirname(settings.tf_model_path), exist_ok=True)
    model.save(settings.tf_model_path)
    with open(settings.tokenizer_path, "wb") as f:
        pickle.dump(encoder, f)
    logger.info("Saved model to %s and label encoder to %s", settings.tf_model_path, settings.tokenizer_path)


if __name__ == "__main__":
    build_and_train_classifier()
