# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
"""
Prepares raw data and provides training and test datasets.
"""

import argparse
from pathlib import Path
import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import mlflow

def parse_args():
    '''Parse input arguments'''

    parser = argparse.ArgumentParser("prep")
    parser.add_argument("--raw_data", type=str, help="Path to raw data")
    parser.add_argument("--train_data", type=str, help="Path to train dataset")
    parser.add_argument("--test_data", type=str, help="Path to test dataset")
    parser.add_argument("--test_train_ratio", type=float, default=0.2, help="Test-train ratio")
    args = parser.parse_args()

    return args

def main(args):
    '''Read, preprocess, split, and save datasets'''

    # Reading Data
    df = pd.read_csv(args.raw_data)

    # Drop irrelevant columns
    if "CustomerID" in df.columns:
        logging.info("Dropping irrelevant column: CustomerID")
        df = df.drop(["CustomerID"], axis=1)
        

    # One-hot encoding for categorical columns
    categorical_columns = df.select_dtypes(include=["object"]).columns
    
    if len(categorical_columns) > 0:
        logging.info(f"Applying one-hot encoding to columns: {list(categorical_columns)}")

        encoder = OneHotEncoder(sparse=False, drop="first")  # Avoid dummy variable trap
        
        encoded_data = pd.DataFrame(
            encoder.fit_transform(df[categorical_columns]),
            columns=encoder.get_feature_names_out(categorical_columns),
        )

        df = pd.concat([df.drop(categorical_columns, axis=1), encoded_data], axis=1)
        

    # Log the first few rows of the transformed dataframe
    logging.info(f"Transformed Data:\n{df.head()}")

    # Split data into train and test
    train_df, test_df = train_test_split(df, test_size=args.test_train_ratio, random_state=42)

    # Save train and test data
    os.makedirs(args.train_data, exist_ok=True)
    os.makedirs(args.test_data, exist_ok=True)
    train_df.to_csv(os.path.join(args.train_data, "train.csv"), index=False)
    test_df.to_csv(os.path.join(args.test_data, "test.csv"), index=False)

    # Log the metrics
    mlflow.log_metric('train size', train_df.shape[0])
    mlflow.log_metric('test size', test_df.shape[0])

if __name__ == "__main__":
    mlflow.start_run()

    # Parse Arguments
    args = parse_args()

    lines = [
        f"Raw data path: {args.raw_data}",
        f"Train dataset output path: {args.train_data}",
        f"Test dataset path: {args.test_data}",
        f"Test-train ratio: {args.test_train_ratio}",
    ]

    for line in lines:
        print(line)
    
    main(args)
    mlflow.end_run()
