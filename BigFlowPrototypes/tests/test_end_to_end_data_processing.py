import unittest
import pandas as pd
import numpy as np
import tempfile
import os
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import time

from BigFlowPrototypes.Linear import main

from BigFlowPrototypes.Linear_Embeded import main as bigflow_main  # Adjust the import to match your actual path

class TestMainFunction(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        # Load the Titanic dataset directly from a URL
        cls.df = pd.read_csv('https://web.stanford.edu/class/archive/cs/cs109/cs109.1166/stuff/titanic.csv')

    def test_main(self):
        # Select a subset of columns for testing
        df = self.df[['Name', 'Age', 'Survived']].dropna()  # Ensure there are no missing values

        with tempfile.NamedTemporaryFile(delete=False, suffix='.csv') as temp_file:
            df.to_csv(temp_file.name, index=False)
            temp_file_path = temp_file.name

        try:
            # Define input parameters
            input_data_path = temp_file_path
            label_column = 'Survived'
            columns_to_embed = ['Name']

            # Run the BigFlow main function
            X_train, X_test, y_train, y_test = bigflow_main(input_data_path, label_column, columns_to_embed)

            # Print shapes of DataFrames
            print("Shape of X_train:", X_train.shape)
            print("Shape of X_test:", X_test.shape)
            print("Shape of y_train:", y_train.shape)
            print("Shape of y_test:", y_test.shape)

            # Assertions
            self.assertIsNotNone(X_train)
            self.assertIsNotNone(X_test)
            self.assertIsNotNone(y_train)
            self.assertIsNotNone(y_test)

            # Expected number of features after PCA should be 5
            expected_num_features = 5
            self.assertEqual(X_train.shape[1], expected_num_features, f"Expected {expected_num_features} features after PCA, but got {X_train.shape[1]}")
            self.assertEqual(X_test.shape[1], expected_num_features)
            self.assertEqual(len(X_train), len(y_train))
            self.assertEqual(len(X_test), len(y_test))

        finally:
            os.remove(temp_file_path)

    def test_process_data(self):
        # Select a subset of columns for testing
        df = self.df[['Name', 'Age', 'Survived']].dropna()  # Ensure there are no missing values

        with tempfile.NamedTemporaryFile(delete=False, suffix='.csv') as temp_file:
            df.to_csv(temp_file.name, index=False)
            temp_file_path = temp_file.name

        try:
            # Define input parameters
            input_file = temp_file_path
            label = 'Survived'

            # Run the process_data function
            X_train, X_test, y_train, y_test, X, y = process_data(input_file, label)

            # Print shapes of DataFrames
            print("Shape of X_train:", X_train.shape)
            print("Shape of X_test:", X_test.shape)
            print("Shape of y_train:", y_train.shape)
            print("Shape of y_test:", y_test.shape)

            # Assertions
            self.assertIsNotNone(X_train)
            self.assertIsNotNone(X_test)
            self.assertIsNotNone(y_train)
            self.assertIsNotNone(y_test)
            self.assertEqual(X_train.shape[1], X_test.shape[1])  # Ensure same number of features
            self.assertEqual(len(X_train), len(y_train))
            self.assertEqual(len(X_test), len(y_test))
            self.assertEqual(X_train.shape[1], 1)  # Because only 'Age' remains after dropping non-numeric columns

        finally:
            os.remove(temp_file_path)
    def test_linear_main(self):
            # Select a subset of columns for testing
            df = self.df[['Name', 'Age', 'Survived']].dropna()  # Ensure there are no missing values

            with tempfile.NamedTemporaryFile(delete=False, suffix='.csv') as temp_file:
                df.to_csv(temp_file.name, index=False)
                temp_file_path = temp_file.name

            try:
                # Define input parameters
                input_data_path = temp_file_path
                label_column = 'Survived'
                columns_to_embed = ['Name']

                # Run the Linear main function
                X_train, X_test, y_train, y_test = linear_main(input_data_path, label_column, columns_to_embed)

                # Print shapes of DataFrames
                print("Shape of X_train:", X_train.shape)
                print("Shape of X_test:", X_test.shape)
                print("Shape of y_train:", y_train.shape)
                print("Shape of y_test:", y_test.shape)

                # Assertions
                self.assertIsNotNone(X_train)
                self.assertIsNotNone(X_test)
                self.assertIsNotNone(y_train)
                self.assertIsNotNone(y_test)

                # Expected number of features after PCA should be 5
                expected_num_features = 5
                self.assertEqual(X_train.shape[1], expected_num_features, f"Expected {expected_num_features} features after PCA, but got {X_train.shape[1]}")
                self.assertEqual(X_test.shape[1], expected_num_features)
                self.assertEqual(len(X_train), len(y_train))
                self.assertEqual(len(X_test), len(y_test))

            finally:
                os.remove(temp_file_path)

if __name__ == '__main__':
    unittest.main()
