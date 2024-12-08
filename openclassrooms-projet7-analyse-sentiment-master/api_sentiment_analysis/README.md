api_sentiment_analysis/
├── app/
│ ├── **init**.py # Initialise l'application Flask
│ ├── routes.py # Contient les routes Flask
│ ├── models.py # Chargement des modèles MLflow et DistilBERT
│ ├── utils.py # Fonctions utilitaires (calcul des embeddings, prétraitement)
│ ├── config.py # Configuration de l'application
│ └── templates/ # Contient les fichiers HTML si nécessaire
│ └── index.html # Exemple de formulaire HTML pour prédiction
├── tests/ # Tests unitaires de l'API
│ ├── **init**.py # Initialise le package de tests
│ ├── test_routes.py # Tests pour les routes de l'API
│ ├── test_models.py # Tests pour les modèles (MLflow/DistilBERT)
│ ├── test_utils.py # Tests pour les fonctions utilitaires
├── run.py # Fichier principal pour exécuter l'API
├── requirements.txt # Dépendances Python nécessaires
└── README.md # Documentation du projet

# Env azure déploiement api 
antenv
