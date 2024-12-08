import mlflow
import mlflow.sklearn
from lightgbm import LGBMClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import classification_report, accuracy_score
from sklearn.model_selection import train_test_split, GridSearchCV
import pandas as pd

# Set MLflow tracking URI and experiment name
mlflow.set_tracking_uri("./mlruns")
mlflow.set_experiment("Sentiment Analysis GridSearch")

# Load the dataset
df_sentiment = pd.read_csv("dataset/dataset_sentiment_clean.csv")
X = df_sentiment["text_clean"]
y = df_sentiment["target"]

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Define TF-IDF vectorizer
tfidf_vectorizer = TfidfVectorizer(max_features=5000)

# Define parameter grid for LightGBM
param_grid = {
    "num_leaves": [15, 31, 63],
    "max_depth": [5, 10, 15],
    "learning_rate": [0.01, 0.1, 0.3],
    "n_estimators": [50, 100, 200],
}

# TF-IDF transformation
print("Transforming data with TF-IDF...")
X_train_transformed = tfidf_vectorizer.fit_transform(X_train)
X_test_transformed = tfidf_vectorizer.transform(X_test)

# Perform GridSearch with MLflow
print("Running GridSearch for TF-IDF...")
with mlflow.start_run(run_name="GridSearch_TFIDF_LightGBM"):
    # Perform GridSearch
    grid_search = GridSearchCV(
        LGBMClassifier(random_state=42),
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
    mlflow.log_dict(classification_rep, "TFIDF_LightGBM_classification_report.json")

    # Log the model
    mlflow.sklearn.log_model(
        sk_model=grid_search.best_estimator_,
        artifact_path="TFIDF_LightGBM",
        registered_model_name="TFIDF_LightGBM",
    )

    print("Completed GridSearch for TF-IDF")
    print(f"Best Parameters: {best_params}")
    print(f"Accuracy: {accuracy * 100:.2f}%")
