from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
import pandas as pd
import mlflow
import mlflow.sklearn
import lightgbm as lgb
from sklearn.metrics import classification_report, accuracy_score

# Set MLflow tracking URI and experiment name
mlflow.set_tracking_uri("./mlruns")
mlflow.set_experiment("Sentiment Analysis Models")

# Load data with pandas
df_sentiment = pd.read_csv("dataset/dataset_sentiment_clean.csv")
X = df_sentiment["text_clean"]
y = df_sentiment["target"]

# Split the data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Define models and their hyperparameters
models = {
    "LogisticRegression": {
        "model": LogisticRegression(max_iter=1000, random_state=42),
        "params": {"max_iter": 1000, "solver": "lbfgs"},
    },
    "RandomForestClassifier": {
        "model": RandomForestClassifier(
            n_estimators=100, max_depth=10, random_state=42
        ),
        "params": {"n_estimators": 100, "max_depth": 10},
    },
    "LightGBM": {
        "model": lgb.LGBMClassifier(
            num_leaves=31,
            max_depth=10,
            learning_rate=0.1,
            n_estimators=100,
            random_state=42,
        ),
        "params": {
            "num_leaves": 31,
            "max_depth": 10,
            "learning_rate": 0.1,
            "n_estimators": 100,
        },
    },
}

# TFIDF parameters
tfidf_params = {"max_features": [5000], "ngram_range": [(1, 1)]}

# Loop through models and TFIDF parameters
for max_features in tfidf_params["max_features"]:
    for ngram_range in tfidf_params["ngram_range"]:
        # Create TFIDF vectorizer with specific parameters
        tfidf_vectorizer = TfidfVectorizer(
            max_features=max_features, ngram_range=ngram_range
        )
        X_train_tfidf = tfidf_vectorizer.fit_transform(X_train)
        X_test_tfidf = tfidf_vectorizer.transform(X_test)

        for model_name, model_dict in models.items():
            model = model_dict["model"]
            model_params = model_dict["params"]

            with mlflow.start_run(
                run_name=f"{model_name}_TFIDF_{max_features}_{ngram_range}"
            ):
                # Log TFIDF parameters
                mlflow.log_param("tfidf_max_features", max_features)
                mlflow.log_param("tfidf_ngram_range", ngram_range)

                # Log model-specific parameters
                for param_name, param_value in model_params.items():
                    mlflow.log_param(param_name, param_value)

                # Train the model
                model.fit(X_train_tfidf, y_train)

                # Evaluate the model
                y_pred = model.predict(X_test_tfidf)
                accuracy = accuracy_score(y_test, y_pred)

                # Log metrics
                mlflow.log_metric("accuracy", accuracy)

                # Log the model
                mlflow.sklearn.log_model(
                    sk_model=model,
                    artifact_path=f"{model_name}_model",
                    registered_model_name=f"{model_name}_tfidf",
                )

                # log the classification report
                classification_rep = classification_report(
                    y_test, y_pred, output_dict=True
                )
                mlflow.log_dict(
                    classification_rep, f"{model_name}_classification_report.json"
                )
