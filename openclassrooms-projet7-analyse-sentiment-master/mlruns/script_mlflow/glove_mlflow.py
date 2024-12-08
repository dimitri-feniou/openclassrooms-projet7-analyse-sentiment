from sklearn.model_selection import train_test_split
import pandas as pd
import mlflow
import mlflow.sklearn
import lightgbm as lgb
from sklearn.metrics import classification_report, accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.decomposition import PCA
import numpy as np

# Set MLflow tracking URI and experiment name
mlflow.set_tracking_uri("./mlruns")
mlflow.set_experiment("Sentiment Analysis Models")

# Load data
df_sentiment = pd.read_csv("dataset/dataset_sentiment_clean.csv")


def load_glove_model(glove_file):
    print("Loading GloVe model...")
    glove_model = {}
    with open(glove_file, "r", encoding="utf-8") as file:
        for line in file:
            split_line = line.split()
            word = split_line[0]
            vector = np.array(split_line[1:], dtype="float32")
            glove_model[word] = vector
    print(f"Loaded {len(glove_model)} words.")
    return glove_model


def compute_avg_glove_vector(texts, glove_model, vector_size):
    """Compute the average GloVe vector for a list of texts."""
    vectors = []
    for text in texts:
        words = text.split()
        word_vectors = [glove_model[word] for word in words if word in glove_model]
        if len(word_vectors) > 0:
            vectors.append(np.mean(word_vectors, axis=0))
        else:
            vectors.append(np.zeros(vector_size))
    return np.array(vectors)


glove_size = 300
glove_model = load_glove_model("model/glove.6B.300d.txt")

# Calculer les vecteurs GloVe moyens pour chaque texte
X = compute_avg_glove_vector(df_sentiment["text_clean"], glove_model, glove_size)
y = df_sentiment["target"]

# Réduire les dimensions avec PCA
reduced_size = 100  # Réduire à 100 dimensions
pca = PCA(n_components=reduced_size, random_state=42)
X_reduced = pca.fit_transform(X)

# Diviser les données en ensembles d'entraînement et de test
X_train, X_test, y_train, y_test = train_test_split(
    X_reduced, y, test_size=0.2, random_state=42
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
    with mlflow.start_run(run_name=f"{model_name}_glove"):
        # Log glove parameters
        mlflow.log_param("glove_size", glove_size)
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
            registered_model_name=f"{model_name}_glove",
        )

        # Log the classification report
        classification_rep = classification_report(y_test, y_pred, output_dict=True)
        mlflow.log_dict(classification_rep, f"{model_name}_classification_report.json")

        # Print summary
        print(f"Model: {model_name}")
        print(f"Accuracy: {accuracy * 100:.2f}%")
        print("Classification Report:")
        print(classification_rep)
