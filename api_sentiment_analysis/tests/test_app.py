import pytest
from unittest.mock import patch, MagicMock
from app import create_app

@pytest.fixture
def app():
    """Create and configure a new app instance for each test."""
    app = create_app({"TESTING": True})
    return app

@pytest.fixture
def client(app):
    """A test client for the app."""
    return app.test_client()

@pytest.fixture
def mock_load_model():
    """Mock the load_model function"""
    with patch('app.models.load_model') as mock:
        mock.return_value = (
            MagicMock(),  # distilbert_model
            MagicMock(),  # tokenizer
            MagicMock(),  # mlflow_model
            MagicMock()   # pca_model
        )
        yield mock

def test_home_page(client):
    """Test the home page route"""
    response = client.get('/')
    assert response.status_code == 200
    assert b'Analyse de Sentiment' in response.data

def test_feedback_route(client):
    """Test the feedback route"""
    response = client.post('/feedback', data={'feedback': 'like', 'text': 'Test feedback'})
    assert response.status_code == 302  # Redirect after successful feedback

def test_sentiment_analysis_success(client, mock_load_model):
    """Test successful sentiment analysis"""
    with patch('app.utils.predict_sentiment') as mock_predict:
        mock_predict.return_value = 1  # Simule un sentiment positif
        response = client.post('/', data={'text': 'This is a test'})
        assert response.status_code == 200
        assert b'positif' in response.data

def test_sentiment_analysis_empty_text(client):
    """Test sentiment analysis with empty text"""
    response = client.post('/', data={'text': ''})
    assert response.status_code == 200
    assert b'Please enter some text to analyze' in response.data

def test_sentiment_analysis_error(client):
    """Test sentiment analysis with error"""
    with patch('app.routes.distilbert_model', None), \
         patch('app.routes.tokenizer', None), \
         patch('app.routes.mlflow_model', None), \
         patch('app.routes.pca_model', None):
        response = client.post('/', data={'text': 'This is a test'})
        assert response.status_code == 200
        assert b'An error occurred during prediction' in response.data
