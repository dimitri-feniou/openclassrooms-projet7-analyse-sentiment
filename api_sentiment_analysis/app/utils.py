import numpy as np
import tensorflow as tf


def predict_sentiment(text, tokenizer, model, mlflow_model, pca_model, max_len=128):
    """
    Compute DistilBERT embedding for a single text, reduce dimensions using PCA, and predict sentiment.

    Args:
        text (str): Input text for sentiment analysis.
        tokenizer: Pre-trained tokenizer (e.g., DistilBERT).
        model: Pre-trained model (e.g., DistilBERT).
        mlflow_model: MLflow-loaded model for prediction.
        max_len (int): Maximum length for text tokenization.

    Returns:
        int: Predicted sentiment label.
    """
    # Tokenize and prepare inputs
    inputs = tokenizer(
        text,
        truncation=True,
        padding="max_length",
        max_length=max_len,
        return_tensors="tf",
    )

    # Validation des dimensions
    input_ids_shape = inputs["input_ids"].shape
    if input_ids_shape[-1] > max_len:
        raise ValueError("Input text is too long")

    # Pass inputs through the model
    output = model(inputs)
    embedding = tf.reduce_mean(output.last_hidden_state, axis=1).numpy()

    # Reduce dimensions using PCA
    reduced_embedding = pca_model.transform(embedding)

    # Predict sentiment using the MLflow model
    prediction = mlflow_model.predict(reduced_embedding)

    return prediction[0]
