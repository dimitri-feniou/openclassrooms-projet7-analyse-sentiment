from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
import pandas as pd
import numpy as np
import mlflow
import mlflow.sklearn
from transformers import DistilBertTokenizer, DistilBertModel
import torch
import lightgbm as lgb
from sklearn.decomposition import PCA
import math
from tqdm import tqdm
from sklearn.decomposition import IncrementalPCA
import joblib

# Set MLflow tracking URI and experiment name
mlflow.set_tracking_uri("./mlruns")
mlflow.set_experiment("Sentiment Analysis Models")

# Set device for computations
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# Load the dataset
df_sentiment = pd.read_csv("dataset/dataset_sentiment_clean.csv")
df_sampled = df_sentiment.sample(frac=0.5, random_state=42)
# Load distilbert tokenizer and model
distilbert_model_name = "distilbert-base-uncased"
tokenizer = DistilBertTokenizer.from_pretrained(distilbert_model_name)
model = DistilBertModel.from_pretrained(distilbert_model_name).to(device)


# Function to compute embeddings with progress tracking
def get_distilbert_embeddings(texts, batch_size=10, max_len=128):
    """
    Returns mean vector for the last hidden layer from the distilbert model.

    Args:
        texts: list[str] - List of sentences/paragraphs
        batch_size: int - Number of sentences in a single batch

    Returns:
        torch.tensor -- Embedding for each sentence/paragraph in texts
    """
    model.eval()
    num_buckets = math.ceil(len(texts) / batch_size)
    vectors = []

    with torch.no_grad():
        # Wrap bucket processing with tqdm for progress tracking
        for bucket in tqdm(
            np.array_split(texts, num_buckets), desc="Computing distilbert Embeddings"
        ):
            tokens = tokenizer(
                bucket.tolist(),
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=max_len,
            ).to(device)
            embeddings = model(**tokens).last_hidden_state.mean(dim=1).detach().cpu()
            vectors.append(embeddings)

    return torch.cat(vectors)


# Compute embeddings with progress bar
print("Computing distilbert embeddings in batches...")
bert_embeddings = get_distilbert_embeddings(
    df_sampled["text_clean"].tolist(), batch_size=8
)
# Reduce dimensions using PCA
print("Reducing dimensions using PCA...")
reduced_size = 100
incremental_pca = IncrementalPCA(n_components=100, batch_size=128)
X_reduced = incremental_pca.fit_transform(bert_embeddings)
joblib.dump(incremental_pca, "pca_model.pkl")
print("PCA model saved to pca_model.pkl")
# Prepare data for training and testing
y = df_sampled["target"].astype(np.int8)
X_train, X_test, y_train, y_test = train_test_split(
    X_reduced, y, test_size=0.2, random_state=42, stratify=df_sampled["target"]
)

# Define models
models = {
    "LogisticRegression": LogisticRegression(max_iter=1000, random_state=42),
    "RandomForestClassifier": RandomForestClassifier(
        n_estimators=100, max_depth=10, random_state=42
    ),
    "LightGBM": lgb.LGBMClassifier(
        num_leaves=31,
        max_depth=10,
        learning_rate=0.1,
        n_estimators=100,
        random_state=42,
    ),
}

# Train and log each model
for model_name, clf in models.items():
    with mlflow.start_run(run_name=f"{model_name}_distilbert"):
        # Train the model
        clf.fit(X_train, y_train)

        # Predict and evaluate
        y_pred = clf.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)

        # Log metrics and model
        mlflow.log_param("pca_n_components", 100)
        mlflow.log_metric("accuracy", accuracy)
        mlflow.sklearn.log_model(
            sk_model=clf,
            artifact_path=f"{model_name}_model",
            registered_model_name=f"{model_name}_distilbert",
        )

        # Log classification report
        classification_rep = classification_report(y_test, y_pred, output_dict=True)
        mlflow.log_dict(classification_rep, f"{model_name}_classification_report.json")

        # Print results
        print(f"Model: {model_name}")
        print(f"Accuracy: {accuracy * 100:.2f}%")
        print("Classification Report:")
        print(classification_rep)
