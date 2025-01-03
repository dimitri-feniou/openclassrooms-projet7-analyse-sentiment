from flask import (
    Blueprint,
    request,
    jsonify,
    render_template,
    current_app,
    url_for,
    redirect,
)
from app.models import load_model
from app.utils import predict_sentiment
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
# Get the absolute path to the templates directory
TEMPLATE_FOLDER = os.path.abspath(os.path.join(os.path.dirname(__file__), "templates"))

# Init blueprint
api = Blueprint("api", __name__)

# Debug print to verify template folder
print(f"Template Folder Path: {TEMPLATE_FOLDER}")
print(f"Template Folder Exists: {os.path.exists(TEMPLATE_FOLDER)}")
print(f"Template Folder Contents: {os.listdir(TEMPLATE_FOLDER)}")

# Load models
try:
    distilbert_model, tokenizer, mlflow_model, pca_model = load_model()
    print("Models loaded successfully")
except Exception as e:
    print(f"Error loading models: {e}")
    distilbert_model = tokenizer = mlflow_model = pca_model = None

@api.route("/", methods=["GET", "POST"])
def home():
    template_folder = current_app.template_folder or TEMPLATE_FOLDER
    print("Flask App Template Folder:", template_folder)

    if request.method == "POST":
        text = request.form.get("text")
        if not text:
            return render_template(
                "index.html", error="Please enter some text to analyze."
            )

        try:
            # Vérifier que les modèles sont chargés
            if None in (distilbert_model, tokenizer, mlflow_model, pca_model):
                raise Exception("Models not loaded properly")

            # Utiliser la fonction predict_sentiment locale
            sentiment = predict_sentiment(
                text, tokenizer, distilbert_model, mlflow_model, pca_model
            )
            sentiment_label = "positif" if sentiment == 1 else "negatif"
            return render_template("index.html", text=text, sentiment=sentiment_label,predicted_sentiment=sentiment_label)
        except Exception as e:
            print(f"Error during prediction: {e}")
            return render_template(
                "index.html", error="An error occurred during prediction."
            )

    return render_template("index.html", text=None, sentiment=None)

@api.route("/feedback", methods=["POST"])
def feedback():
    feedback = request.form.get("feedback")  # 'like' ou 'dislike'
    text = request.form.get("text")          # text to predic
    predicted_sentiment = request.form.get("predicted_sentiment")  # predict sentiment
    # Debug print result from form
    print(f"Feedback reçu : {feedback}")
    print(f"Texte : {text}")
    print(f"Sentiment prédit : {predicted_sentiment}")

    # save the telemetry azure app insigth
    from applicationinsights import TelemetryClient
    tc = TelemetryClient('e0a1e652-439b-440b-a8bd-c6996203174b')
    tc.track_event(
        'FeedbackReceived',
        properties={
            'feedback': feedback,
            'text': text,
            'predicted_sentiment': predicted_sentiment
        }
    )
    tc.flush()

    return redirect("/")

