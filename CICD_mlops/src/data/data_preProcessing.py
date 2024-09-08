import subprocess
import sys

def install(package):
    subprocess.check_call([sys.executable, "-m", "pip", "install", package])


install("nltk")
install("pandas")
install("numpy")
# install('string')
install('wordnet')
install('stopwords')

import pandas as pd # type: ignore
import numpy as np # type: ignore
import re
import os
import nltk # type: ignore
import string
from nltk.corpus import stopwords # type: ignore
from nltk.stem import SnowballStemmer, WordNetLemmatizer # type: ignore
from sklearn.feature_extraction.text import CountVectorizer # type: ignore
import logging

# Logging configuration
logger = logging.getLogger('data pre-processing')
logger.setLevel('DEBUG')

console_handler = logging.StreamHandler()
console_handler.setLevel('DEBUG')

file_handler = logging.FileHandler('data_pre-processing.log')
file_handler.setLevel('ERROR')

formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
console_handler.setFormatter(formatter)
file_handler.setFormatter(formatter)

logger.addHandler(console_handler)
logger.addHandler(file_handler)

# Transform the data
# nltk.download('wordnet')
# nltk.download('stopwords')

def lemmatization(text: str) -> str:
    try:
        lemmatizer = WordNetLemmatizer()
        text = text.split()
        text = [lemmatizer.lemmatize(word) for word in text]
        return " ".join(text)
    except Exception as e:
        logger.error(f"An error occurred: {e}")
        return ""

def remove_stop_words(text: str) -> str:
    try:
        stop_words = set(stopwords.words("english"))
        text = [i for i in str(text).split() if i not in stop_words]
        return " ".join(text)
    except Exception as e:
        logger.debug(f"An error occurred in remove_stop_words: {e}")
        return text  # Return the original text if an error occurs

def removing_numbers(text: str) -> str:
    try:
        text = ''.join([i for i in text if not i.isdigit()])
        return text
    except Exception as e:
        logger.debug(f"An error occurred in removing_numbers: {e}")
        return text  # Return the original text if an error occurs

def lower_case(text: str) -> str:
    try:
        text = text.split()
        text = [y.lower() for y in text]
        return " ".join(text)
    except Exception as e:
        print(f"An error occurred in lower_case: {e}")
        return text  # Return the original text if an error occurs

def removing_punctuations(text: str) -> str:
    try:
        # Remove punctuations
        text = re.sub('[%s]' % re.escape("""!"#$%&'()*+,،-./:;<=>؟?@[\]^_`{|}~"""), ' ', text)
        text = text.replace('؛', "", )
        # Remove extra whitespace
        text = re.sub('\s+', ' ', text)
        text = " ".join(text.split())
        return text.strip()
    except Exception as e:
        logger.debug(f"An error occurred in removing_punctuations: {e}")
        return text  # Return the original text if an error occurs

def removing_urls(text: str) -> str:
    try:
        url_pattern = re.compile(r'https?://\S+|www\.\S+')
        return url_pattern.sub(r'', text)
    except Exception as e:
        logger.debug(f"An error occurred in removing_urls: {e}")
        return text  # Return the original text if an error occurs

def remove_small_sentences(df: pd.DataFrame) -> None:
    try:
        for i in range(len(df)):
            if len(df.text.iloc[i].split()) < 3:
                df.text.iloc[i] = np.nan
    except Exception as e:
        print(f"An error occurred in remove_small_sentences: {e}")

def normalize_text(df: pd.DataFrame) -> pd.DataFrame:
    try:
        df['content'] = df['content'].apply(lambda content: lower_case(content))
        df['content'] = df['content'].apply(lambda content: remove_stop_words(content))
        df['content'] = df['content'].apply(lambda content: removing_numbers(content))
        df['content'] = df['content'].apply(lambda content: removing_punctuations(content))
        df['content'] = df['content'].apply(lambda content: removing_urls(content))
        df['content'] = df['content'].apply(lambda content: lemmatization(content))
    except Exception as e:
        print(f"An error occurred during text normalization: {e}")
    return df

# store the data inside data/processed
def clean_data(train_data: pd.DataFrame, test_data: pd.DataFrame) -> pd.DataFrame:
    try:
        # Fill missing values in both train and test datasets
        train_data.fillna('', inplace=True)
        test_data.fillna('', inplace=True)
        
        # Return the cleaned datasets
        return train_data, test_data
    except Exception as e:
        logger.debug(f"An error occurred during data cleaning: {e}")
        return train_data, test_data  # Return the original datasets if an error occurs

def processed_data(train_data: pd.DataFrame, test_data: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    try:
        # Process the train and test data
        train_processed_data = normalize_text(train_data)
        test_processed_data = normalize_text(test_data)
        
        return train_processed_data, test_processed_data
    except Exception as e:
        logger.debug(f"An error occurred during data processing: {e}")
        # Return the original data if an error occurs
        return train_data, test_data

def save_data(data_path: str, train_processed_data: pd.DataFrame, test_processed_data: pd.DataFrame) -> None:
    # data_path = os.path.join("data","processed")
    os.makedirs(data_path)
    train_processed_data.to_csv(os.path.join(data_path,"train_processed.csv"))
    test_processed_data.to_csv(os.path.join(data_path,"test_prccessed.csv"))

# Store the data into data/preprocessed
def main():
    try:
        # Fetch the data from data/raw
        train_data = pd.read_csv('./data/raw/train.csv')
        test_data = pd.read_csv('./data/raw/test.csv')
        
        # Data path
        data_path = os.path.join("data", "processed")
        
        # Clean the data
        train_data, test_data = clean_data(train_data, test_data)
        
        # Process the data
        train_processed_data, test_processed_data = processed_data(train_data, test_data)
        
        # Save the processed data
        save_data(data_path, train_processed_data, test_processed_data)
        
        logger.debug("Data processing and saving completed successfully.")
        
    except Exception as e:
        logger.error(f"An error occurred in the main function: {e}")

if __name__ == "__main__":
    main()
    