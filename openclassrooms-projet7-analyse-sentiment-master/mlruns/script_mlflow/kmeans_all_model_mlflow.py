from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import classification_report, accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.decomposition import IncrementalPCA
import pandas as pd
import mlflow
import mlflow.sklearn
from gensim.models import Word2Vec
from transformers import DistilBertTokenizer, DistilBertModel
import torch
from tqdm import tqdm
import numpy as np
import joblib

# Set MLflow tracking URI and experiment name
mlflow.set_tracking_uri("./mlruns")
mlflow.set_experiment("Sentiment Analysis Models")

# Load the dataset
df_sentiment = pd.read_csv("dataset/dataset_sentiment_clean.csv")
df_sampled = df_sentiment.sample(frac=0.5, random_state=42)
X = df_sampled["text_clean"]
y = df_sampled["target"]


# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)


# Load Word2Vec model (replace with your pre-trained model path if needed)
word2vec_model = Word2Vec.load("models/word2vec_model.bin")

# Load DistilBERT tokenizer and model
distilbert_model_name = "distilbert-base-uncased"
distilbert_tokenizer = DistilBertTokenizer.from_pretrained(distilbert_model_name)
distilbert_model = DistilBertModel.from_pretrained(distilbert_model_name).to(
    "cuda" if torch.cuda.is_available() else "cpu"
)


# Function to compute Word2Vec embeddings
def compute_word2vec_embeddings(texts, model):
    embeddings = []
    for text in texts:
        words = text.split()
        word_vectors = [model.wv[word] for word in words if word in model.wv]
        if word_vectors:
            embeddings.append(np.mean(word_vectors, axis=0))
        else:
            embeddings.append(np.zeros(model.vector_size))
    return np.array(embeddings)


# Function to compute DistilBERT embeddings
def compute_distilbert_embeddings(texts, tokenizer, model, batch_size=8, max_len=128):
    model.eval()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    vectors = []
    with torch.no_grad():
        for batch in tqdm(
            np.array_split(texts, len(texts) // batch_size),
            desc="DistilBERT Embeddings",
        ):
            tokens = tokenizer(
                list(batch),
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=max_len,
            ).to(device)
            output = (
                model(**tokens).last_hidden_state.mean(dim=1).detach().cpu().numpy()
            )
            vectors.extend(output)
    return np.array(vectors)


# Feature extractors
feature_extractors = {
    "TF-IDF": TfidfVectorizer(max_features=5000),
    "Word2Vec": lambda texts: compute_word2vec_embeddings(texts, word2vec_model),
    "DistilBERT": lambda texts: compute_distilbert_embeddings(
        texts,
        distilbert_tokenizer,
        distilbert_model,
        batch_size=8,
    ),
}

# KMeans parameters
kmeans_params = {"n_clusters": 2, "random_state": 42}

# Test each feature extractor with KMeans
for feature_name, feature_extractor in feature_extractors.items():
    if feature_name == "TF-IDF":
        X_train_features = feature_extractor.fit_transform(X_train)
        X_test_features = feature_extractor.transform(X_test)
    else:
        X_train_features = feature_extractor(X_train.tolist())
        X_test_features = feature_extractor(X_test.tolist())

    # Reduce dimensions for Word2Vec and DistilBERT if necessary
    if feature_name != "TF-IDF" and X_train_features.shape[1] > 100:
        pca = IncrementalPCA(n_components=50)
        X_train_features = pca.fit_transform(X_train_features)
        X_test_features = pca.transform(X_test_features)

    # Train KMeans model
    with mlflow.start_run(run_name=f"KMeans_{feature_name}"):
        kmeans = KMeans(**kmeans_params)
        kmeans.fit(X_train_features)

        # Predict clusters
        y_pred = kmeans.predict(X_test_features)

        # Log metrics
        accuracy = accuracy_score(y_test, y_pred)
        mlflow.log_param("n_clusters", kmeans_params["n_clusters"])
        mlflow.log_metric("accuracy", accuracy)

        # Log the model
        mlflow.sklearn.log_model(
            sk_model=kmeans,
            artifact_path=f"KMeans_{feature_name}_model",
            registered_model_name=f"KMeans_{feature_name}_classification",
        )
        if feature_name != "TF-IDF":
            joblib.dump(pca, f"pca_{feature_name}.pkl")
            mlflow.log_artifact(f"pca_{feature_name}.pkl")
        # Log classification report
        classification_rep = classification_report(y_test, y_pred, output_dict=True)
        mlflow.log_dict(
            classification_rep, f"KMeans_{feature_name}_classification_report.json"
        )

        # Print results
        print(f"Feature: {feature_name}")
        print(f"Accuracy: {accuracy * 100:.2f}%")
        print("Classification Report:")
        print(classification_rep)
