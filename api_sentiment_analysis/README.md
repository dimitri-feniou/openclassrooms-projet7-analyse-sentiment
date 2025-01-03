# Sentiment Analysis API

## Project Structure 

```
api_sentiment_analysis/
│
├── app/                         # Application core directory
│   ├── __init__.py              # Flask app initialization
│   ├── routes.py                # Flask routes definition
│   ├── models.py                # MLflow and DistilBERT models loading
│   ├── utils.py                 # Utility functions (embeddings, preprocessing)
│   ├── config.py                # Application configuration
│   └── templates/               # HTML templates directory
│       └── index.html           # HTML form example for predictions
│
├── tests/                       # Unit tests directory
│   ├── __init__.py             # Test package initialization
│   ├── test_routes.py          # API routes tests
│   ├── test_models.py          # Models tests (MLflow/DistilBERT)
│   └── test_utils.py           # Utility functions tests
│
├── run.py                      # Main API execution file
├── requirements.txt            # Python dependencies
├── requirements-test.txt       # Python dependencies for testing
├── runtime.txt                 # Python runtime version
├── startup.sh                  # Startup script for deployment
└── README.md                   # Project documentation
```

## Running the Flask App (local)

To run the Flask application, follow these steps:

1. **Create a virtual environment**:
    ```sh
    python -m venv venv
    ```

2. **Activate the virtual environment**:
    - On Windows:
        ```sh
        venv\Scripts\activate
        ```
    - On macOS/Linux:
        ```sh
        source venv/bin/activate
        ```

3. **Install the dependencies**:
    ```sh
    pip install -r requirements.txt
    ```

4. **Run the Flask application**:
    ```sh
    python run.py
    ```

The application will be available at `http://127.0.0.1:8000`.

## Running the Tests (local)

To run the tests, follow these steps:

1. **Create a virtual environment** (if not already created):
    ```sh
    python -m venv venv
    ```

2. **Activate the virtual environment**:
    - On Windows:
        ```sh
        venv\Scripts\activate
        ```
    - On macOS/Linux:
        ```sh
        source venv/bin/activate
        ```

3. **Install the test dependencies**:
    ```sh
    pip install -r requirements-test.txt
    ```

4. **Run the tests**:
    ```sh
    pytest
    ```

The tests will be executed, and the results will be displayed in the terminal.