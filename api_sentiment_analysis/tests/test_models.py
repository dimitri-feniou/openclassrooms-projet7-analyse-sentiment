import pytest
from unittest.mock import patch, MagicMock
import numpy as np
import tensorflow as tf
import os
import sys

# Ajouter le chemin de la racine du projet au sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../")))

from app.models import load_model

class MockMLflowModel:
    def __init__(self):
        self.n_features_in_ = 100

    def predict(self, X):
        if X.shape[1] != self.n_features_in_:
            raise ValueError(f"X has {X.shape[1]} features, but model is expecting {self.n_features_in_} features")
        # Simule une prédiction binaire
        return np.array([1])

@pytest.fixture
def mock_all_models():
    """Fixture pour mocker tous les modèles"""
    mock_tokenizer = MagicMock()
    mock_tokenizer.return_value = {
        "input_ids": tf.constant([[1, 2, 3]]),
        "attention_mask": tf.constant([[1, 1, 1]]),
    }
    
    mock_model = MagicMock()
    mock_model.return_value = MagicMock(
        last_hidden_state=tf.constant([[[0.1, 0.2, 0.3]]])
    )
    
    mock_pca = MagicMock()
    mock_pca.transform.return_value = np.array([[0.5] * 100])  # Retourne 100 features
    
    mock_mlflow = MockMLflowModel()
    
    return mock_tokenizer, mock_model, mock_pca, mock_mlflow

def test_load_model(mock_all_models):
    """Test le chargement des modèles"""
    mock_tokenizer, mock_model, mock_pca, mock_mlflow = mock_all_models
    
    with patch('transformers.DistilBertTokenizer.from_pretrained', return_value=mock_tokenizer), \
         patch('transformers.TFDistilBertModel.from_pretrained', return_value=mock_model), \
         patch('mlflow.sklearn.load_model', return_value=mock_mlflow), \
         patch('joblib.load', return_value=mock_pca):
        
        # Test le chargement des modèles
        distilbert_model, tokenizer, mlflow_model, pca_model = load_model()
        
        # Vérifie que chaque modèle est chargé
        assert distilbert_model is not None, "DistilBERT model not loaded"
        assert tokenizer is not None, "Tokenizer not loaded"
        assert mlflow_model is not None, "MLflow model not loaded"
        assert pca_model is not None, "PCA model not loaded"
        
        # Test la fonctionnalité de chaque modèle
        # Test tokenizer
        tokens = tokenizer("Test text")
        assert "input_ids" in tokens, "Tokenizer failed to process text"
        assert "attention_mask" in tokens, "Tokenizer failed to create attention mask"
        
        # Test DistilBERT
        outputs = distilbert_model(tokens)
        assert hasattr(outputs, 'last_hidden_state'), "DistilBERT failed to process tokens"
        
        # Test PCA
        test_data = np.random.randn(1, 768)  # DistilBERT output dimension
        pca_output = pca_model.transform(test_data)
        assert pca_output.shape == (1, 100), f"PCA output shape incorrect: {pca_output.shape}"
        
        # Test MLflow model avec un mock personnalisé
        test_input = pca_output  # Utilise la sortie du PCA comme entrée
        prediction = mlflow_model.predict(test_input)
        assert isinstance(prediction, np.ndarray), "MLflow model prediction should be numpy array"
        assert prediction.shape == (1,), f"MLflow model prediction shape incorrect: {prediction.shape}"
        assert prediction[0] in [0, 1], f"MLflow model prediction not binary: {prediction[0]}"
