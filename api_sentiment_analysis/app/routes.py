from flask import Blueprint, request, jsonify, render_template, current_app, url_for,redirect
from app.models import load_model
from app.utils import predict_sentiment
import os

# Get the absolute path to the templates directory
TEMPLATE_FOLDER = os.path.abspath(os.path.join(os.path.dirname(__file__), "templates"))

# Init blueprint
api = Blueprint(
    "api", __name__, template_folder=TEMPLATE_FOLDER  # Use the absolute path
)

# Debug print to verify template folder
print(f"Template Folder Path: {TEMPLATE_FOLDER}")
print(f"Template Folder Exists: {os.path.exists(TEMPLATE_FOLDER)}")
print(f"Template Folder Contents: {os.listdir(TEMPLATE_FOLDER)}")

# Load model from distilbert and
distilbert_model, tokenizer, mlflow_model, pca_model = load_model()


@api.route("/", methods=["GET", "POST"])
def home():
    # More robust template folder checking
    template_folder = current_app.template_folder or TEMPLATE_FOLDER

    print("Flask App Template Folder:", template_folder)

    # Check if template folder exists before listing
    if os.path.exists(template_folder):
        print("Template Files:", os.listdir(template_folder))
    else:
        print(f"Template directory does not exist: {template_folder}")
    if request.method == "POST":
        text = request.form.get("text")
        if not text:
            return render_template(
                "index.html", error="Please enter some text to analyze."
            )

        try:
            sentiment = predict_sentiment(
                text, tokenizer, distilbert_model, mlflow_model, pca_model
            )
            sentiment_label = "positif" if sentiment == 1 else "negatif"
        except Exception as e:
            print(f"Error during prediction: {e}")
            return render_template(
                "index.html", error="An error occurred during prediction."
            )
        print(sentiment_label)
        return render_template("index.html", text=text, sentiment=sentiment_label)

    return render_template("index.html", text=None, sentiment=None)


@api.route("/feedback", methods=["POST"])
def feedback():
    feedback = request.form.get("feedback")  # 'like' or 'dislike'
    text = request.form.get("text")  # The analyzed text
    print(f"Feedback received: {feedback} for text: {text}")
    # Enregistrez le feedback dans une base de données ou un fichier
    return redirect("/")
