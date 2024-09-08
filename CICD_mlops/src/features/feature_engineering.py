# import subprocess
# import sys

# def install(package):
#     subprocess.check_call([sys.executable, "-m", "pip", "install", package])


# install("nltk")
# install("pandas")
# install("numpy")
# # install('string')
# install('wordnet')
# install('stopwords')

import pandas as pd # type: ignore
import numpy as np # type: ignore
import re
import os
from sklearn.feature_extraction.text import CountVectorizer # type: ignore
import yaml # type: ignore
import logging

# Logging configuration
logger = logging.getLogger('feature-engineering')
logger.setLevel('DEBUG')

console_handler = logging.StreamHandler()
console_handler.setLevel('DEBUG')

file_handler = logging.FileHandler('feature_engineering.log')
file_handler.setLevel('ERROR')

formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
console_handler.setFormatter(formatter)
file_handler.setFormatter(formatter)

logger.addHandler(console_handler)
logger.addHandler(file_handler)

# Import params.yaml file
max_features = yaml.safe_load(open('params.yaml','r'))['feature_engineering']['max_features']

def process_features(train_data: pd.DataFrame, test_data:  pd.DataFrame) ->  pd.DataFrame:
    # Fetch data from data/processed
    # train_data = pd.read_csv('./data/processed/train_processed.csv')
    # test_data = pd.read_csv('./data/processed/test_prccessed.csv')

    train_data.fillna('', inplace=True)
    test_data.fillna('', inplace=True)
    logger.debug("cleaning completed")

    X_train = train_data['content'].values
    y_train = train_data['sentiment'].values
    logger.debug("trained data completed")

    X_test = test_data['content'].values
    y_test = test_data['sentiment'].values
    logger.debug("test data completed")

    # Apply Bag of Words (CountVectorizer)
    vectorizer = CountVectorizer(max_features=max_features)
    logger.debug("bag of words applied")

    # Fit the vectorizer on the training data and transform it
    X_train_bow = vectorizer.fit_transform(X_train)

    # Transform the test data using the same vectorizer
    X_test_bow = vectorizer.transform(X_test)

    train_df = pd.DataFrame(X_train_bow.toarray())
    train_df['label'] = y_train

    test_df = pd.DataFrame(X_test_bow.toarray())
    test_df['label'] = y_test
    logger.debug("labelling completed for test and train")
    return train_df, test_df
    
def save_data_feature(data_path: str, train_df:  pd.DataFrame, test_df:  pd.DataFrame) -> None:

# store the data into data/features
    os.makedirs(data_path)
    train_df.to_csv(os.path.join(data_path,"train_bow.csv"))
    test_df.to_csv(os.path.join(data_path,"test_bow.csv"))
    
def main():
    train_data = pd.read_csv('./data/processed/train_processed.csv')
    test_data = pd.read_csv('./data/processed/test_prccessed.csv')
    data_path = os.path.join("data","feature")
    logger.error("reading data and path is completed")
    
    train_df, test_df = process_features(train_data, test_data)
    save_data_feature(data_path, train_df, test_df)
    logger.error("save the data completed")
    
if __name__ == "__main__":
    main()
    
    