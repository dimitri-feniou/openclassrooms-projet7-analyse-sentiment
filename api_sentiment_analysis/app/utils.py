import torch
import numpy as np


def predict_sentiment(text, tokenizer, model, mlflow_model, pca_model,max_len=128):
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
    if len(text) > max_len * 10:  # Limitez selon vos besoins
        raise ValueError("Text is too long for processing")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.eval()

    # Tokenize the text and compute the embedding
    with torch.no_grad():
        encoded_input = tokenizer(
            text,
            padding="max_length",
            truncation=True,
            max_length=max_len,
            return_tensors="pt",
        ).to(device)
        output = model(**encoded_input)
        embedding = output.last_hidden_state.mean(dim=1).detach().cpu().numpy()

    reduced_embedding = pca_model.transform(embedding)

    # Predict sentiment using the MLflow model
    prediction = mlflow_model.predict(reduced_embedding)

    return prediction[0]
