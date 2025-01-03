import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
import lightgbm as lgb
from sklearn.metrics import accuracy_score, classification_report, precision_recall_fscore_support
import mlflow
import mlflow.sklearn

# Set up MLflow experiment
mlflow.set_experiment("Sentiment_analysis_model_test")

# Load data
df_sentiment = pd.read_csv("/home/semoulolait/Documents/openclassrooms/projet_7/dataset/dataset_sentiment_clean.csv")

def prepare_data():
    # Prepare features and target
    X = df_sentiment["text_clean"]
    y = df_sentiment["target"]
    
    # TFIDF Vectorization
    vectorizer = TfidfVectorizer(max_features=5000)
    X_vectorized = vectorizer.fit_transform(X)
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X_vectorized, y, test_size=0.2, random_state=42
    )
    return X_train, X_test, y_train, y_test, vectorizer

# Define models to test
models = {
    "LogisticRegression": LogisticRegression(max_iter=1000),
    "RandomForest": RandomForestClassifier(n_estimators=100, max_depth=10),
    "LightGBM": lgb.LGBMClassifier(num_leaves=31, max_depth=10, learning_rate=0.1, n_estimators=100)
}

# Prepare data
X_train, X_test, y_train, y_test, vectorizer = prepare_data()

# Train and evaluate each model
for model_name, model in models.items():
    with mlflow.start_run(run_name=f"TFIDF_{model_name}"):
        # Log model parameters
        mlflow.log_params(model.get_params())
        
        # Train model
        model.fit(X_train, y_train)
        
        # Make predictions
        y_pred = model.predict(X_test)
        
        # Calculate metrics
        accuracy = accuracy_score(y_test, y_pred)
        
        # Get classification report metrics
        precision, recall, f1, support = precision_recall_fscore_support(y_test, y_pred)
        
        # Log metrics
        mlflow.log_metric("accuracy", accuracy)
        for i, label in enumerate(["negative", "positive"]):
            mlflow.log_metric(f"precision_{label}", precision[i])
            mlflow.log_metric(f"recall_{label}", recall[i])
            mlflow.log_metric(f"f1_{label}", f1[i])
            mlflow.log_metric(f"support_{label}", support[i])
        
        # Log classification report as text artifact
        report = classification_report(y_test, y_pred)
        with open("classification_report.txt", "w") as f:
            f.write(report)
        mlflow.log_artifact("classification_report.txt")
        
        # Log model
        mlflow.sklearn.log_model(model, f"tfidf_{model_name.lower()}_model")
        
        # Add model type tag
        mlflow.set_tag("model_type", f"tfidf_{model_name.lower()}")
