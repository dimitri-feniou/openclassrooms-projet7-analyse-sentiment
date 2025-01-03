import os
import math
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
import lightgbm as lgb
from sklearn.metrics import accuracy_score, classification_report, precision_recall_fscore_support
import mlflow
import mlflow.sklearn
import torch
from transformers import DistilBertTokenizer, DistilBertModel
from sklearn.decomposition import IncrementalPCA
from tqdm import tqdm
import gc

# Set up MLflow experiment
mlflow.set_experiment("Sentiment_analysis_model_test")

# Set device for computations
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Load the dataset
print("Loading data...")
df_sentiment = pd.read_csv("/home/semoulolait/Documents/openclassrooms/projet_7/dataset/dataset_sentiment_clean.csv")

# Load distilbert tokenizer and model
distilbert_model_name = "distilbert-base-uncased"
tokenizer = DistilBertTokenizer.from_pretrained(distilbert_model_name)
model = DistilBertModel.from_pretrained(distilbert_model_name).to(device)

# Function to compute embeddings with progress tracking and memory optimization
def get_distilbert_embeddings(texts, batch_size=10, max_len=128):
    model.eval()
    num_buckets = math.ceil(len(texts) / batch_size)
    vectors = []

    with torch.no_grad():
        for bucket in tqdm(np.array_split(texts, num_buckets), desc="Computing distilbert Embeddings"):
            tokens = tokenizer(
                bucket.tolist(),
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=max_len,
            ).to(device)
            
            embeddings = model(**tokens).last_hidden_state.mean(dim=1).detach().cpu()
            vectors.append(embeddings)
            
            # Clear GPU memory and release unused RAM
            del tokens, embeddings
            torch.cuda.empty_cache()
            if torch.cuda.is_available():
                torch.cuda.synchronize()

    return torch.cat(vectors)

# Compute embeddings with progress bar
print("Computing distilbert embeddings in batches...")
bert_embeddings = get_distilbert_embeddings(
    df_sentiment["text_clean"].tolist(), batch_size=128
)

# Reduce dimensions using Incremental PCA
print("Reducing dimensions using PCA...")
reduced_size = 100
incremental_pca = IncrementalPCA(n_components=reduced_size, batch_size=64)

# First pass: fit IncrementalPCA
for i in range(0, len(bert_embeddings), 1000):
    chunk = bert_embeddings[i:i + 1000]
    incremental_pca.partial_fit(chunk)

# Second pass: transform data
X_reduced = []
for i in range(0, len(bert_embeddings), 1000):
    chunk = bert_embeddings[i:i + 1000]
    X_reduced.append(incremental_pca.transform(chunk))
X_reduced = np.vstack(X_reduced)

# Free memory
del bert_embeddings, incremental_pca
torch.cuda.empty_cache()
gc.collect()

# Prepare data for training and testing
y = df_sentiment["target"].astype(np.int8)
X_train, X_test, y_train, y_test = train_test_split(
    X_reduced, y, test_size=0.2, random_state=42
)

# Define models to test
models = {
    "LogisticRegression": LogisticRegression(max_iter=1000, random_state=42),
    "RandomForest": RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42),
    "LightGBM": lgb.LGBMClassifier(num_leaves=31, max_depth=10, learning_rate=0.1, n_estimators=100, random_state=42)
}

# Train and evaluate each model
for model_name, model in models.items():
    print(f"Training and evaluating model: {model_name}")
    with mlflow.start_run(run_name=f"DistilBERT_{model_name}"):
        mlflow.log_params(model.get_params())
        model.fit(X_train, y_train)

        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        precision, recall, f1, support = precision_recall_fscore_support(y_test, y_pred)

        mlflow.log_metric("accuracy", accuracy)
        for i, label in enumerate(["negative", "positive"]):
            mlflow.log_metric(f"precision_{label}", precision[i])
            mlflow.log_metric(f"recall_{label}", recall[i])
            mlflow.log_metric(f"f1_{label}", f1[i])
            mlflow.log_metric(f"support_{label}", support[i])

        report = classification_report(y_test, y_pred)
        with open("classification_report.txt", "w") as f:
            f.write(report)
        mlflow.log_artifact("classification_report.txt")

        mlflow.sklearn.log_model(model, f"distilbert_{model_name.lower()}_model")
        mlflow.set_tag("model_type", f"distilbert_{model_name.lower()}")

print("Script completed.")