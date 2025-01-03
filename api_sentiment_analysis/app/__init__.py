import os
from flask import Flask
from applicationinsights.flask.ext import AppInsights

def create_app(config=None):
    """
    Factory function to create and configure the Flask app.
    """
    app = Flask(
        __name__,
        template_folder=os.path.join(os.path.dirname(__file__), "templates"),
    )

    # Ajouter la clé directement ici
    APPINSIGHTS_INSTRUMENTATIONKEY = "e0a1e652-439b-440b-a8bd-c6996203174b"
    app.config["APPINSIGHTS_INSTRUMENTATIONKEY"] = APPINSIGHTS_INSTRUMENTATIONKEY

    # Initialiser AppInsights pour l'application principale
    appinsights = AppInsights(app)

    # Importer et enregistrer le blueprint
    from app.routes import api
    app.register_blueprint(api)

    return app
