import unittest
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.linear_model import LinearRegression
import os
import subprocess

# Function to download dataset from Kaggle
def download_kaggle_dataset(dataset, path):
    subprocess.run(['kaggle', 'datasets', 'download', '-d', dataset, '--unzip', '-p', path], check=True)

# Simple data processing function
def simple_process_data(input_file, label):
    df = pd.read_csv(input_file)
    non_numeric_columns = [col for col in df.columns if df[col].dtype == 'object']
    df.drop(columns=non_numeric_columns, inplace=True)
    df = df.fillna(df.mean(numeric_only=True))
    X = df.drop(columns=[label])
    y = df[label]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    return X_train, X_test, y_train, y_test

# Advanced data processing function
def advanced_process_data(input_file, label, columns_to_embed):
    df = pd.read_csv(input_file)
    for col in columns_to_embed:
        embedded_df = embed_column_openai(df, col)
        df = pd.concat([df, embedded_df], axis=1)
    df.drop(columns=columns_to_embed, inplace=True)
    non_numeric_columns = df.select_dtypes(exclude=[np.number]).columns.tolist()
    df.drop(columns=non_numeric_columns, inplace=True)
    df = df.fillna(df.mean(numeric_only=True))
    X = df.drop(columns=[label])
    y = df[label]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    return X_train, X_test, y_train, y_test

class TestDataProcessingComparison(unittest.TestCase):
    
    @classmethod
    def setUpClass(cls):
        # Download the Titanic dataset from Kaggle
        cls.dataset_path = 'titanic'
        download_kaggle_dataset('heptapod/titanic', cls.dataset_path)
        cls.input_file = os.path.join(cls.dataset_path, 'train.csv')
        cls.label = 'Survived'

    def test_simple_process_data(self):
        X_train, X_test, y_train, y_test = simple_process_data(self.input_file, self.label)
        self.assertEqual(X_train.shape[0], int(0.8 * len(pd.read_csv(self.input_file))))
        self.assertEqual(X_test.shape[0], int(0.2 * len(pd.read_csv(self.input_file))))
        self.assertEqual(y_train.shape[0], int(0.8 * len(pd.read_csv(self.input_file))))
        self.assertEqual(y_test.shape[0], int(0.2 * len(pd.read_csv(self.input_file))))

    def test_advanced_process_data(self):
        columns_to_embed = ['Name', 'Ticket', 'Cabin']  # Example text columns to embed
        X_train, X_test, y_train, y_test = advanced_process_data(self.input_file, self.label, columns_to_embed)
        self.assertEqual(X_train.shape[0], int(0.8 * len(pd.read_csv(self.input_file))))
        self.assertEqual(X_test.shape[0], int(0.2 * len(pd.read_csv(self.input_file))))
        self.assertEqual(y_train.shape[0], int(0.8 * len(pd.read_csv(self.input_file))))
        self.assertEqual(y_test.shape[0], int(0.2 * len(pd.read_csv(self.input_file))))

    def test_comparison(self):
        X_train_simple, X_test_simple, y_train_simple, y_test_simple = simple_process_data(self.input_file, self.label)
        columns_to_embed = ['Name', 'Ticket', 'Cabin']  # Example text columns to embed
        X_train_advanced, X_test_advanced, y_train_advanced, y_test_advanced = advanced_process_data(self.input_file, self.label, columns_to_embed)
        
        # Fit a simple model to both datasets and compare results
        lr_simple = LinearRegression().fit(X_train_simple, y_train_simple)
        lr_advanced = LinearRegression().fit(X_train_advanced, y_train_advanced)
        
        mse_simple = mean_squared_error(y_test_simple, lr_simple.predict(X_test_simple))
        mse_advanced = mean_squared_error(y_test_advanced, lr_advanced.predict(X_test_advanced))
        
        r2_simple = r2_score(y_test_simple, lr_simple.predict(X_test_simple))
        r2_advanced = r2_score(y_test_advanced, lr_advanced.predict(X_test_advanced))
        
        print(f"Simple Processing - MSE: {mse_simple}, R2: {r2_simple}")
        print(f"Advanced Processing - MSE: {mse_advanced}, R2: {r2_advanced}")
        
        self.assertLessEqual(mse_advanced, mse_simple)
        self.assertGreaterEqual(r2_advanced, r2_simple)

if __name__ == '__main__':
    unittest.main()
