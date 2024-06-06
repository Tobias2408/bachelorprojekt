import sys
import os
import site

# Add the site packages path to sys.path
site_packages_path = "/Library/Frameworks/Python.framework/Versions/3.10/lib/python3.10/site-packages"
sys.path.append(site_packages_path)

# Now try importing
from preprocess_ai.DataPreparation.DataCleaner.dataCleaner import DataCleaner
from preprocess_ai.DataPreparation.DataCleaner.duplicationHandler import DuplicationHandler
from preprocess_ai.DataPreparation.DataCleaner.dataTransformer import DataTransformer
from preprocess_ai.DataPreparation.DataTransformation.categoricalEncoder import CategoricalEncoder
from preprocess_ai.DataPreparation.DataReduction.dimensionalityReducer import DimensionalityReducer
from preprocess_ai.DataPreparation.DataReduction.featureSelector import FeatureSelector

# Sample code to demonstrate data processing using preprocess_ai package

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import time

# Timing function
def timeit(method):
    def timed(*args, **kwargs):
        start_time = time.time()
        result = method(*args, **kwargs)
        end_time = time.time()
        print(f"{method.__name__} took {end_time - start_time:.2f} seconds")
        return result
    return timed

@timeit
def process_data(input_file, label):
    print("Loading data...")
    df = pd.read_csv(input_file)

    # Identify non-numeric columns
    non_numeric_columns = [col for col in df.columns if df[col].dtype == 'object']

    # Drop non-numeric columns
    df.drop(columns=non_numeric_columns, inplace=True)

    # Handle missing values
    df = df.fillna(df.mean(numeric_only=True))

    # Splitting data
    X = df.drop(columns=[label])
    y = df[label]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Standardize the data
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    
    return X_train, X_test, y_train, y_test, X, y

@timeit
def main(input_data_path, label_column, columns_to_embed):
    start_total = time.time()
    
    print("Loading data...")
    df = pd.read_csv(input_data_path)

    label = label_column

    # Embed specified text columns
    embedded_columns = []
    for col in columns_to_embed:
        embedded_df = embed_column_openai(df, col)
        df = pd.concat([df, embedded_df], axis=1)
        embedded_columns.extend(embedded_df.columns)
    df.drop(columns=columns_to_embed, inplace=True)

    # Remove all non-numeric, non-embedded columns
    non_numeric_columns = df.select_dtypes(exclude=[np.number]).columns.tolist()
    df.drop(columns=non_numeric_columns, inplace=True)

    # Clean the data
    print("Cleaning data...")
    cleaner = DataCleaner()
    cleaned_data = cleaner.smart_clean_data(df, label, missing_threshold=0.3, correlation_threshold=0.5)

    # Handle duplicates
    print("Handling duplicates...")
    handler = DuplicationHandler()
    cleaned_data = handler.remove_duplicates(cleaned_data)

    # Data transformation
    print("Transforming data...")
    transformer = DataTransformer()
    numeric_columns = cleaned_data.select_dtypes(include=[np.number]).columns.tolist()
    if label in numeric_columns:
        numeric_columns.remove(label)

    for column in numeric_columns:
        skewness = cleaned_data[column].skew()
        if abs(skewness) > 0.5:
            cleaned_data = transformer.apply_log_transformation(cleaned_data, column)

    for column in numeric_columns:
        Q1 = cleaned_data[column].quantile(0.25)
        Q3 = cleaned_data[column].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        outliers = ((cleaned_data[column] < lower_bound) | (cleaned_data[column] > upper_bound)).sum()
        if outliers > 0:
            cleaned_data = transformer.impute_outliers(cleaned_data, column, method='median')

        cleaned_data = transformer.winsorize_data(cleaned_data, column, limits=(0.05, 0.05))

    # Remove columns that are entirely NaN
    cleaned_data = cleaned_data.dropna(axis=1, how='all')

    # Ensure consistent columns before imputation
    consistent_columns = cleaned_data.columns.tolist()

    # Impute missing values
    print("Imputing missing values...")
    imputer = SimpleImputer(strategy='mean')
    transformed_data = imputer.fit_transform(cleaned_data)
    transformed_df = pd.DataFrame(transformed_data, columns=consistent_columns)

    # Ensure no NaN values are present
    if transformed_df.isnull().values.any():
        print("NaN values found in the data after imputation. Performing imputation again.")
        transformed_df = pd.DataFrame(imputer.fit_transform(transformed_df), columns=consistent_columns)

    # Feature selection using SelectKBest
    print("Selecting features...")
    X = transformed_df.drop(columns=[label])
    y = transformed_df[label]
    
    feature_selector = FeatureSelector()
    selected_df = feature_selector.select_k_best(X, y, k=10, score_func=f_regression)
    selected_df[label] = y

    # Dimensionality reduction using PCA
    print("Reducing dimensionality...")
    reducer = DimensionalityReducer()
    reduced_df, pca_model, scaler = reducer.fit_transform_pca(selected_df.drop(columns=[label]), n_components=5)
    reduced_df[label] = selected_df[label]

    # Split the data into train and test sets
    print("Splitting data into train and test sets...")
    X = reduced_df.drop(columns=[label])
    y = reduced_df[label]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Normalize the data
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    return X_train, X_test, y_train, y_test

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process input data for AI models.")
    parser.add_argument('input_data_path', type=str, help='Path to the input data file (e.g., titanic.csv)')
    parser.add_argument('label_column', type=str, help='Name of the label column (e.g., Survived)')
    parser.add_argument('columns_to_embed', nargs='+', help='Columns to embed (e.g., Name)')
    
    args = parser.parse_args()
    
    X_train, X_test, y_train, y_test = main(args.input_data_path, args.label_column, args.columns_to_embed)
    
    print("Data processing complete.")
