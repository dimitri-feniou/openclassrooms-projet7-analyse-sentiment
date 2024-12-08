import os
import pytest


@pytest.fixture(scope="session", autouse=True)
def set_working_directory():
    # Définit le dossier de travail comme la racine du projet
    os.chdir(os.path.abspath(os.path.dirname(__file__)) + "/..")
