import pandas as pd
import numpy as np
from pathlib import Path
import logging
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from check_structure import check_existing_file, check_existing_folder
import os

def main():
    """ Runs data processing scripts to turn raw data from (../raw) into
        cleaned data ready to be analyzed (saved in../preprocessed).
    """
    logger = logging.getLogger(__name__)
    logger.info('making final data set from raw data')

    filepath = "data/processed"
    process_data(filepath)

def process_data(filepath):
    # Import dataset
    X_train = import_dataset(f"{filepath}/X_train.csv", sep=",")
    X_test = import_dataset(f"{filepath}/X_test.csv", sep=",")

    X_train, X_test =  normalize_data(X_train, X_test)

    # Save dataframes to their respective output file paths
    save_dataframes(X_train, X_test, filepath)

def import_dataset(file_path, **kwargs):
    return pd.read_csv(file_path, **kwargs)

def normalize_data(X_train, X_test):
    # use standardscaler for normalization
    scaler = StandardScaler().fit(X_train)

    # Transform and wrap as DataFrame
    X_train_scaled = pd.DataFrame(
        scaler.transform(X_train), 
        index=X_train.index,   # keep original index
        columns=X_train.columns  # keep original column names
    )

    X_test_scaled = pd.DataFrame(
        scaler.transform(X_test), 
        index=X_test.index, 
        columns=X_test.columns
    )

    return X_train_scaled, X_test_scaled


def save_dataframes(X_train, X_test, output_folderpath):
    # Save dataframes to their respective output file paths
    for file, filename in zip([X_train, X_test], ['X_train_scaled', 'X_test_scaled']):
        output_filepath = os.path.join(output_folderpath, f'{filename}.csv')
        if check_existing_file(output_filepath):
            file.to_csv(output_filepath, index=False) 

if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    main()