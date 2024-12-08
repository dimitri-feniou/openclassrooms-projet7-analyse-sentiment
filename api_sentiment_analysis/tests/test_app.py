import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))


import pytest
import torch
from unittest.mock import MagicMock, patch
from app.models import load_model
from app.utils import predict_sentiment
from flask import Flask
from app.routes import api


def test_template_debug(client):
    import os

    print("Current Working Directory:", os.getcwd())
    print("Template Path:", client.application.jinja_loader.searchpath)


@pytest.fixture
def client():
    """Create a test client for the Flask application"""
    from flask import Flask
    from app.routes import TEMPLATE_FOLDER, api

    app = Flask(__name__, template_folder=TEMPLATE_FOLDER)
    app.config["TESTING"] = True
    app.register_blueprint(api)

    # Debug print
    print(f"Test Client Template Folder: {app.template_folder}")
    print(f"Template Folder Exists: {os.path.exists(app.template_folder)}")

    return app.test_client()


@pytest.fixture
def mock_models():
    """Fixture to mock the loaded models"""
    # Mock tokenizer
    mock_tokenizer = MagicMock()
    mock_encoded_input = {
        "input_ids": torch.tensor([[1, 2, 3, 4, 5]]),
        "attention_mask": torch.tensor([[1, 1, 1, 1, 1]]),
    }
    mock_tokenizer.return_value = type(
        "MockEncodedInput", (), {"to": lambda device: mock_encoded_input}
    )

    # Mock DistilBERT model
    mock_distilbert_model = MagicMock()
    mock_last_hidden_state = torch.tensor([[[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]]])
    mock_output = MagicMock()
    mock_output.last_hidden_state = mock_last_hidden_state
    mock_distilbert_model.__call__ = lambda *args, **kwargs: mock_output
    mock_distilbert_model.eval = lambda: None

    # Mock MLflow model
    mock_mlflow_model = MagicMock()
    mock_mlflow_model.predict.return_value = [1]  # Positive sentiment

    # Mock PCA model
    mock_pca_model = MagicMock()
    mock_pca_model.transform.return_value = [[0.1, 0.2]]

    return mock_distilbert_model, mock_tokenizer, mock_mlflow_model, mock_pca_model


def test_predict_sentiment(mock_models):
    """Test the sentiment prediction utility function"""
    distilbert_model, tokenizer, mlflow_model, pca_model = mock_models

    text = "I love this product!"
    prediction = predict_sentiment(
        text, tokenizer, distilbert_model, mlflow_model, pca_model
    )

    assert prediction == 1, "Prediction should be positive"


def test_home_route_get(client):
    import os

    response = client.get("/")
    assert response.status_code == 200

    # Check for key elements from the template
    assert b"Analyse de Sentiment" in response.data  # Check for the main title
    assert (
        b"Entrez un texte ci-dessous pour commencer" in response.data
    )  # Check for description
    assert b'<textarea name="text"' in response.data  # Verify textarea exists
    assert b"Analyze" in response.data  # Check for analyze button
    assert (
        b"R\xc3\xa9ecrire un texte" in response.data
    )  # Check for "Rewrite text" button


def test_home_route_post_valid_text(client):
    """Test home route with valid POST text"""
    response = client.post("/", data={"text": "I am happy today!"})
    assert response.status_code == 200

    # Check for sentiment result and key template elements
    assert b"Analyse de Sentiment" in response.data
    assert b"Sentiment:" in response.data
    assert b"positif" in response.data or b"negatif" in response.data


def test_home_route_post_empty_text(client):
    """Test home route with empty text"""
    response = client.post("/", data={"text": ""})
    assert response.status_code == 200
    assert b"Please enter some text to analyze" in response.data


@patch("app.models.load_model")
def test_load_model_cuda_availability(mock_load_model):
    """Test model loading with CUDA availability"""
    with patch("torch.cuda.is_available", return_value=True):
        from app.models import load_model

        load_model()
        mock_load_model.assert_called_once()


def test_predict_sentiment_error_handling(mock_models):
    """Test sentiment prediction with potential error scenarios"""
    distilbert_model, tokenizer, mlflow_model, pca_model = mock_models

    # Test avec un texte trop long
    long_text = "a" * 10000
    with pytest.raises(ValueError, match="Text is too long for processing"):
        predict_sentiment(
            long_text, tokenizer, distilbert_model, mlflow_model, pca_model
        )


def test_model_loading_paths():
    """Verify that model loading paths are correctly configured"""
    from app.config import MODEL_URI, PCA_PATH
    import os

    assert os.path.exists(MODEL_URI), f"Model path {MODEL_URI} does not exist"
    assert os.path.exists(PCA_PATH), f"PCA path {PCA_PATH} does not exist"
