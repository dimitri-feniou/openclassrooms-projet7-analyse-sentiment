import os
from flask import Flask


def create_app(config=None):
    app = Flask(
        __name__,
        template_folder=os.path.join(os.path.dirname(__file__), "templates"),
    )
    if config:
        app.config.update(config)
    from app.routes import api

    app.register_blueprint(api)
    return app
