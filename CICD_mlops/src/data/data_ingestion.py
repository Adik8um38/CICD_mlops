import subprocess
import sys
import os
import pandas as pd # type: ignore
import numpy as np # type: ignore
import yaml # type: ignore
from sklearn.model_selection import train_test_split # type: ignore
import logging

# Logging configuration
logger = logging.getLogger('data ingestion')
logger.setLevel('DEBUG')

console_handler = logging.StreamHandler()
console_handler.setLevel('DEBUG')

file_handler = logging.FileHandler('data_ingestion_error.log')
file_handler.setLevel('ERROR')

formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
console_handler.setFormatter(formatter)
file_handler.setFormatter(formatter)

logger.addHandler(console_handler)
logger.addHandler(file_handler)


def install(package):
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", package])
    except subprocess.CalledProcessError as e:
        # print(f"Error occurred while installing package {package}: {e}")
        logger.error(f'Error occurred while installing package {package}: {e}')
        sys.exit(1)

try:
    install("numpy")
    install("scikit-learn")
    # install("yaml")
except Exception as e:
    # print(f"Installation error: {e}")
    logger.error(f"Installation error: {e}")
    sys.exit(1)

def load_params(param_path: str) -> float:
    try:
        with open(param_path, 'r') as file:
            params = yaml.safe_load(file)
            test_size = params['data_ingestion']['test_size']
            logger.debug("test size retrived")
        return test_size
    except FileNotFoundError:
        # print(f"Parameter file not found: {param_path}")
        logger.error(f"Parameter file not found: {param_path}")
        sys.exit(1)
    except yaml.YAMLError as e:
        # print(f"Error parsing YAML file: {e}")
        logger.error(f"Error parsing YAML file: {e}")
        sys.exit(1)
    except KeyError:
        print("The required key 'data_ingestion' or 'test_size' is missing in the YAML file.")
        sys.exit(1)

def read_data(path: str) -> pd.DataFrame:
    try:
        df = pd.read_csv(path)
        return df
    except FileNotFoundError:
        print(f"Data file not found: {path}")
        sys.exit(1)
    except pd.errors.EmptyDataError:
        print("Data file is empty.")
        sys.exit(1)
    except pd.errors.ParserError:
        print("Error parsing the data file.")
        sys.exit(1)

def process_data(df: pd.DataFrame) -> pd.DataFrame:
    try:
        df.drop(columns=['tweet_id'], inplace=True)
        final_df = df[df['sentiment'].isin(['happiness', 'sadness'])]
        final_df['sentiment'].replace({'happiness': 1, 'sadness': 0}, inplace=True)
        logger.debug('final_df generated')
        return final_df
    except KeyError as e:
        print(f"Column missing in data frame: {e}")
        sys.exit(1)

def save_data(data_path: str, train_data: pd.DataFrame, test_data: pd.DataFrame) -> None:
    try:
        os.makedirs(data_path, exist_ok=True)
        train_data.to_csv(os.path.join(data_path, "train.csv"), index=False)
        test_data.to_csv(os.path.join(data_path, "test.csv"), index=False)
        logger.debug('test & train data saved')
    except OSError as e:
        print(f"Error creating directory or saving files: {e}")
        sys.exit(1)

def main():
    param_path = 'params.yaml'
    path = 'https://raw.githubusercontent.com/campusx-official/jupyter-masterclass/main/tweet_emotions.csv'
    data_path = os.path.join("data", "raw")
    logger.debug('path varaible completed')

    try:
        test_size = load_params(param_path)
        df = read_data(path)
        final_df = process_data(df)
        train_data, test_data = train_test_split(final_df, test_size=test_size, random_state=42)
        save_data(data_path, train_data, test_data)
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
