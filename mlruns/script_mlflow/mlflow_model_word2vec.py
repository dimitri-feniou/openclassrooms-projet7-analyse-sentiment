import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
import lightgbm as lgb
from sklearn.metrics import accuracy_score, classification_report, precision_recall_fscore_support
import mlflow
import mlflow.sklearn
import gensim
from gensim.models import Word2Vec
from sklearn.decomposition import PCA
from gensim.models import KeyedVectors
import os 
# Set up MLflow experiment
mlflow.set_experiment("Sentiment_analysis_model_test")

# Load data
print("Loading data...")
df_sentiment = pd.read_csv("/home/semoulolait/Documents/openclassrooms/projet_7/dataset/dataset_sentiment_clean.csv")

def prepare_data():
    print("Preparing data...")
    # Prepare features and target
    X = df_sentiment["text_clean"]
    y = df_sentiment["target"]
    
    # Word2Vec Vectorization
    print("Training Word2Vec model...")
    # Configuration
    w2v_size = 300
    reduced_size = 100  # Dimension réduite
    model_path = os.path.join(os.path.dirname(__file__), "models", "word2vec_model.bin")
    w2v_model_name = "word2vec_model.bin"
    w2v_model = KeyedVectors.load(w2v_model_name, mmap="r")
    sentences = [gensim.utils.simple_preprocess(text) for text in X]
    
    def compute_avg_word2vec(texts, model, vector_size):
        vectors = []
        for text in texts:
            words = text.split()
            word_vectors = [model.wv[word] for word in words if word in model.wv]
            if len(word_vectors) > 0:
                vectors.append(np.mean(word_vectors, axis=0))
            else:
                vectors.append(np.zeros(vector_size))
        return np.array(vectors)
    
    print("Vectorizing text data...")
    X_vectorized = compute_avg_word2vec(X, w2v_model, 300)
    
    # Reduce dimensions with PCA
    print("Reducing dimensions with PCA...")
    pca = PCA(reduced_size, random_state=42)
    X_reduced = pca.fit_transform(X_vectorized)
    
    # Split data
    print("Splitting data into train and test sets...")
    X_train, X_test, y_train, y_test = train_test_split(
        X_reduced, y, test_size=0.2, random_state=42
    )
    return X_train, X_test, y_train, y_test, w2v_model

# Define models to test
models = {
    "LogisticRegression": LogisticRegression(max_iter=1000),
    "RandomForest": RandomForestClassifier(n_estimators=100, max_depth=10),
    "LightGBM": lgb.LGBMClassifier(num_leaves=31, max_depth=10, learning_rate=0.1, n_estimators=100)
}

# Prepare data
X_train, X_test, y_train, y_test, w2v_model = prepare_data()

# Train and evaluate each model
for model_name, model in models.items():
    print(f"Training and evaluating model: {model_name}")
    with mlflow.start_run(run_name=f"Word2Vec_{model_name}"):
        # Log model parameters
        mlflow.log_params(model.get_params())
        
        # Train model
        print(f"Training {model_name}...")
        model.fit(X_train, y_train)
        
        # Make predictions
        print(f"Making predictions with {model_name}...")
        y_pred = model.predict(X_test)
        
        # Calculate metrics
        accuracy = accuracy_score(y_test, y_pred)
        
        # Get classification report metrics
        precision, recall, f1, support = precision_recall_fscore_support(y_test, y_pred)
        
        # Log metrics
        print(f"Logging metrics for {model_name}...")
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
        print(f"Logging {model_name} model to MLflow...")
        mlflow.sklearn.log_model(model, f"word2vec_{model_name.lower()}_model")
        
        # Add model type tag
        mlflow.set_tag("model_type", f"word2vec_{model_name.lower()}")

print("Script completed.")