import subprocess
import sys
import pandas as pd
import numpy as np
import os
import pickle
import json
import logging

from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_auc_score
import xgboost as xgb

# Configure logging
logger = logging.getLogger()
logger.setLevel(logging.DEBUG)

# Console handler
console_handler = logging.StreamHandler(sys.stdout)
console_handler.setLevel(logging.INFO)
console_formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
console_handler.setFormatter(console_formatter)

# File handler
file_handler = logging.FileHandler('model_evaluation.log')
file_handler.setLevel(logging.DEBUG)
file_formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
file_handler.setFormatter(file_formatter)

# Add handlers to the logger
logger.addHandler(console_handler)
logger.addHandler(file_handler)

def install(package):
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", package])
        logger.info(f"Successfully installed package: {package}")
    except subprocess.CalledProcessError as e:
        logger.error(f"Error occurred while installing package {package}: {e}")
        sys.exit(1)

def load_model(model_path):
    try:
        with open(model_path, 'rb') as file:
            logger.info(f"Successfully loaded model from {model_path}")
            return pickle.load(file)
    except FileNotFoundError:
        logger.error(f"Error: The model file '{model_path}' was not found.")
        sys.exit(1)
    except pickle.UnpicklingError:
        logger.error(f"Error: Failed to unpickle the model file '{model_path}'.")
        sys.exit(1)

def load_data(data_path):
    try:
        data = pd.read_csv(data_path)
        logger.info(f"Successfully loaded data from {data_path}")
        return data
    except FileNotFoundError:
        logger.error(f"Error: The test data file '{data_path}' was not found.")
        sys.exit(1)
    except pd.errors.EmptyDataError:
        logger.error(f"Error: The test data file '{data_path}' is empty.")
        sys.exit(1)
    except pd.errors.ParserError:
        logger.error(f"Error: The test data file '{data_path}' could not be parsed.")
        sys.exit(1)

def evaluate_model(clf, X_test, y_test):
    try:
        y_pred = clf.predict(X_test)
        y_pred_prob = clf.predict_proba(X_test)[:, 1]

        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)
        auc = roc_auc_score(y_test, y_pred_prob)

        logger.info("Model evaluation completed successfully.")
        return {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'auc': auc
        }
    except Exception as e:
        logger.error(f"An error occurred during model evaluation: {e}")
        sys.exit(1)

def save_metrics(metrics, file_path):
    try:
        with open(file_path, 'w') as file:
            json.dump(metrics, file, indent=4)
        logger.info(f"Metrics successfully saved to {file_path}")
    except IOError as e:
        logger.error(f"Error occurred while writing to the file '{file_path}': {e}")
        sys.exit(1)

def main():
    logger.info("Script execution started.")
    
    # Install required packages
    packages = ["nltk", "pandas", "numpy", "xgboost", "corpus"]
    for package in packages:
        install(package)

    # Load the trained model
    clf = load_model('model.pkl')

    # Fetch the test data
    test_data_feature = load_data('./data/feature/test_bow.csv')

    X_test_feature = test_data_feature.iloc[:, 0:-1].values
    y_test_feature = test_data_feature.iloc[:, -1].values

    # Evaluate the model
    metrics = evaluate_model(clf, X_test_feature, y_test_feature)

    # Save the metrics to a JSON file
    save_metrics(metrics, 'metrics.json')

    logger.info("Script execution completed.")

if __name__ == "__main__":
    main()
