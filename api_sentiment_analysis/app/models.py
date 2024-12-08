import mlflow
from transformers import DistilBertTokenizer, DistilBertModel
import torch
from app.config import MODEL_URI
from app.config import PCA_PATH
import joblib
import os

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../models"))
pca_path = os.path.join(BASE_DIR, "pca_model.pkl")
model_uri = os.path.join(BASE_DIR, "mlflow_model/LogisticRegression_model/")
def load_model():
    """
    Load the tokenizer, DistilBERT model, and MLflow sentiment analysis model.
    """
    distilbert_model_name = "distilbert-base-uncased"
    tokenizer = DistilBertTokenizer.from_pretrained(distilbert_model_name)
    distilbert_model = DistilBertModel.from_pretrained(distilbert_model_name).to(
        "cuda" if torch.cuda.is_available() else "cpu"
    )
    mlflow_model = mlflow.sklearn.load_model(model_uri
    )  
    pca_model = joblib.load(pca_path)
    
    return distilbert_model, tokenizer, mlflow_model,pca_model
