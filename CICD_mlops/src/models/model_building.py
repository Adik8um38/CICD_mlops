import subprocess
import sys
import yaml
import pandas as pd
import numpy as np
import pickle
from sklearn.ensemble import GradientBoostingClassifier
import logging

# Configure logging
logger = logging.getLogger('model_building')
logger.setLevel(logging.DEBUG)

# Console handler
console_handler = logging.StreamHandler(sys.stdout)
console_handler.setLevel(logging.DEBUG)
console_formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
console_handler.setFormatter(console_formatter)

# File handler
file_handler = logging.FileHandler('model_building.log')
file_handler.setLevel(logging.ERROR)
file_formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
file_handler.setFormatter(file_formatter)

# Add handlers to logger
logger.addHandler(console_handler)
logger.addHandler(file_handler)

def install(package):
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", package])
        logger.info(f"Successfully installed package: {package}")
    except subprocess.CalledProcessError as e:
        logger.error(f"Error occurred while installing package {package}: {e}")
        sys.exit(1)

# Install required packages
packages = ["nltk", "pandas", "numpy", "wordnet", "stopwords"]
for package in packages:
    install(package)

def read_yaml_file(params_file: str) -> dict:
    try:
        with open(params_file, 'r') as file:
            params = yaml.safe_load(file).get('model_building', {})
            logger.debug("Successfully read YAML file.")
        return params
    except FileNotFoundError:
        logger.error(f"Error: The file {params_file} was not found.")
        return {}
    except yaml.YAMLError as e:
        logger.error(f"Error parsing YAML file: {e}")
        return {}
    except KeyError:
        logger.error("Error: The key 'model_building' was not found in the YAML file.")
        return {}
    except Exception as e:
        logger.error(f"An unexpected error occurred while reading the YAML file: {e}")
        return {}

def data_processed_and_saving_model(params: dict, train_data: pd.DataFrame, test_data: pd.DataFrame) -> None:
    try:
        # Fetch the data
        X_train = train_data.iloc[:, :-1].values
        y_train = train_data.iloc[:, -1].values

        # Initialize and train the model
        clf = GradientBoostingClassifier(n_estimators=params.get('n_estimators', 100), 
                                         learning_rate=params.get('learning_rate', 0.1))
        clf.fit(X_train, y_train)

        # Save the model
        with open('model.pkl', 'wb') as file:
            pickle.dump(clf, file)
        
        logger.debug("Model successfully trained and saved.")

    except KeyError as e:
        logger.error(f"Error: Missing parameter {e} in the YAML file.")
    except FileNotFoundError:
        logger.error("Error: The data files were not found.")
    except pickle.PicklingError:
        logger.error("Error: There was a problem saving the model.")
    except Exception as e:
        logger.error(f"An unexpected error occurred during model processing: {e}")

def main():
    try:
        params_file = "params.yaml"
        train_data = pd.read_csv('./data/feature/train_bow.csv')
        test_data = pd.read_csv('./data/feature/test_bow.csv')
        logger.debug("Data files successfully read.")
        
        params = read_yaml_file(params_file)
        if params:  # Ensure params is not empty
            data_processed_and_saving_model(params, train_data, test_data)
    
    except FileNotFoundError as e:
        logger.error(f"Error: {e}")
    except pd.errors.EmptyDataError:
        logger.error("Error: One of the data files is empty.")
    except Exception as e:
        logger.error(f"An unexpected error occurred in the main function: {e}")

if __name__ == "__main__":
    main()
