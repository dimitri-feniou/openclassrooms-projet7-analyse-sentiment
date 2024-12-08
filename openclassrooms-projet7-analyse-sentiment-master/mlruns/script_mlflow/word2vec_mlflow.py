from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
from sklearn.decomposition import PCA
import pandas as pd
import mlflow
import mlflow.sklearn
import lightgbm as lgb
from gensim.models import KeyedVectors
import numpy as np

# Set MLflow tracking URI and experiment name
mlflow.set_tracking_uri("./mlruns")
mlflow.set_experiment("Sentiment Analysis Models")

# Load data
df_sentiment = pd.read_csv("dataset/dataset_sentiment_clean.csv")

# Load pre-trained Word2Vec model
w2v_model = KeyedVectors.load("model/word2vec_model.bin", mmap="r")

# Configurations
w2v_size = 300
reduced_size = 100


# Function to compute average Word2Vec vectors
def compute_avg_word2vec(df, model, w2v_size):
    vectors = []
    for description in df["text_clean"]:
        words = description.split()
        word_vectors = [model.wv[word] for word in words if word in model.wv]
        if word_vectors:
            vectors.append(np.mean(word_vectors, axis=0))
        else:
            vectors.append(np.zeros(w2v_size))
    return np.array(vectors)


# Compute average Word2Vec vectors
avg_word2vec_vectors = compute_avg_word2vec(df_sentiment, w2v_model, w2v_size)

# Apply PCA to reduce dimensions
pca = PCA(n_components=reduced_size, random_state=42)
reduced_vectors = pca.fit_transform(avg_word2vec_vectors)

# Split the data
X = reduced_vectors
y = df_sentiment["target"].astype(np.int8)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
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

# Start training and logging
for model_name, model in models.items():
    with mlflow.start_run(run_name=f"{model_name}_word2vec"):
        # Log Word2Vec parameters
        mlflow.log_param("word2vec_size", w2v_size)
        mlflow.log_param("pca_reduced_size", reduced_size)

        # Train the model
        model.fit(X_train, y_train)

        # Evaluate the model
        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)

        # Log metrics and model
        mlflow.log_metric("accuracy", accuracy)
        mlflow.sklearn.log_model(
            sk_model=model,
            artifact_path=f"{model_name}_model",
            registered_model_name=f"{model_name}_word2vec",
        )

        # Log the classification report
        classification_rep = classification_report(y_test, y_pred, output_dict=True)
        mlflow.log_dict(classification_rep, f"{model_name}_classification_report.json")

        # Print summary
        print(f"Model: {model_name}")
        print(f"Accuracy: {accuracy * 100:.2f}%")
        print("Classification Report:")
        print(classification_rep)
