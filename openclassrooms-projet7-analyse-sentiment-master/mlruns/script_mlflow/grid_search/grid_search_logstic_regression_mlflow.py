import mlflow
import mlflow.sklearn
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import classification_report, accuracy_score
from sklearn.model_selection import train_test_split, GridSearchCV
from gensim.models import Word2Vec
import pandas as pd
import numpy as np
import torch
from transformers import DistilBertTokenizer, DistilBertModel
from tqdm import tqdm
import joblib
from sklearn.decomposition import IncrementalPCA
import math

# Set MLflow tracking URI and experiment name
mlflow.set_tracking_uri("./mlruns")
mlflow.set_experiment("Sentiment Analysis GridSearch")

# Set device for computations
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load the dataset
df_sentiment = pd.read_csv("dataset/dataset_sentiment_clean.csv")
df_sampled = df_sentiment.sample(frac=0.5, random_state=42)
X = df_sampled["text_clean"]
y = df_sampled["target"]


# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)


# Define function for Word2Vec embeddings
def compute_word2vec_embeddings(texts, word2vec_model):
    embeddings = []
    for text in texts:
        words = text.split()
        vectors = [
            word2vec_model.wv[word] for word in words if word in word2vec_model.wv
        ]
        if len(vectors) > 0:
            embeddings.append(np.mean(vectors, axis=0))
        else:
            embeddings.append(np.zeros(word2vec_model.vector_size))
    return np.array(embeddings)


# Define function for DistilBERT embeddings
def compute_distilbert_embeddings(texts, tokenizer, model, batch_size=8, max_len=128):
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


# Load pre-trained models
word2vec_model = Word2Vec.load("models/word2vec_model.bin")
distilbert_tokenizer = DistilBertTokenizer.from_pretrained("distilbert-base-uncased")
distilbert_model = DistilBertModel.from_pretrained("distilbert-base-uncased").to(device)

# Define feature extraction pipelines
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

# Define parameter grid for LogisticRegression and pca
param_grid = {"penalty": ["l2"], "C": [0.1, 1, 10], "solver": ["lbfgs"]}
pca_params = [50]
# Perform GridSearch with MLflow
for feature_name, feature_extractor in feature_extractors.items():
    print(f"Running GridSearch for {feature_name}...")
    for n_components in pca_params:  # Iterate over PCA components
        with mlflow.start_run(run_name=f"GridSearch_{feature_name}_PCA_{n_components}"):
            if feature_name == "TF-IDF":
                X_train_transformed = feature_extractor.fit_transform(X_train)
                X_test_transformed = feature_extractor.transform(X_test)
            elif feature_name == "DistilBERT":
                # Extract embeddings
                embeddings_train = feature_extractors["DistilBERT"](X_train)
                embeddings_test = feature_extractors["DistilBERT"](X_test)
            else:
                # Extract embeddings
                embeddings_train = feature_extractor(X_train)
                embeddings_test = feature_extractor(X_test)

                # Apply PCA
                if feature_name != "TF-IDF":
                    pca = IncrementalPCA(n_components=n_components)
                    X_train_transformed = pca.fit_transform(embeddings_train)
                    X_test_transformed = pca.transform(embeddings_test)

                # Log PCA parameters
                mlflow.log_param("pca_n_components", n_components)

            # Perform GridSearch
            grid_search = GridSearchCV(
                LogisticRegression(max_iter=1000, random_state=42),
                param_grid,
                cv=3,
                scoring="accuracy",
                n_jobs=-1,
            )
            grid_search.fit(X_train_transformed, y_train)

            # Log best parameters
            best_params = grid_search.best_params_
            mlflow.log_params(best_params)

            # Predict on test data
            y_pred = grid_search.best_estimator_.predict(X_test_transformed)

            # Evaluate and log metrics
            accuracy = accuracy_score(y_test, y_pred)
            classification_rep = classification_report(y_test, y_pred, output_dict=True)
            mlflow.log_metric("accuracy", accuracy)
            if feature_name != "TF-IDF":
                mlflow.log_dict(
                    classification_rep,
                    f"{feature_name}_PCA_{n_components}_classification_report.json",
                )

            # Log the model and PCA
            mlflow.sklearn.log_model(
                sk_model=grid_search.best_estimator_,
                artifact_path=f"{feature_name}_LogisticRegression_{n_components}",
                registered_model_name=f"{feature_name}_LogisticRegression",
            )
            if feature_name != "TF-IDF":
                joblib.dump(pca, f"pca_{feature_name}_{n_components}.pkl")
                mlflow.log_artifact(f"pca_{feature_name}_{n_components}.pkl")

            print(f"Completed GridSearch for {feature_name} with PCA({n_components})")
            print(f"Best Parameters: {best_params}")
            print(f"Accuracy: {accuracy * 100:.2f}%")
