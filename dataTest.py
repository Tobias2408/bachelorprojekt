import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from DataPreparation.DataCleaner.dataCleaner import dataCleaner
from DataPreparation.DataCleaner.duplicationHandler import DuplicationHandler
from DataPreparation.DataCleaner.dataTransformer import DataTransformer
from DataPreparation.DataTransformation.scaler import SimpleScaler
from DataPreparation.DataTransformation.featureEngineer import FeatureEngineer, interaction_feature, date_parts, total_sales
from DataPreparation.DataTransformation.categoricalEncoder import CategoricalEncoder

# Main script
df = pd.read_csv('movies.csv')
label = "Popularity"

# Clean the data
cleaner = dataCleaner()
cleaned_data = cleaner.smart_clean_data(df, label, missing_threshold=0.3, correlation_threshold=0.5)

# Handle duplicates
handler = DuplicationHandler()
cleaned_data = handler.remove_duplicates(cleaned_data)

print("\nDataFrame after Removing Duplicates:")
print(cleaned_data.head())

# Handle categorical data
categorical_columns = cleaned_data.select_dtypes(include=['object']).columns.tolist()
print("\nCategorical Columns Identified (Object Types):")
print(categorical_columns)

# Explicitly handle any columns that might be missed if they contain strings
string_columns = [col for col in cleaned_data.columns if cleaned_data[col].dtype == 'object']
print("\nString Columns Identified:")
print(string_columns)

# Combine categorical and string columns to ensure all categorical data is handled
categorical_columns = list(set(categorical_columns + string_columns))

# Apply one-hot encoding for all categorical columns with a category limit
max_categories = 10
if categorical_columns:
    cleaned_data, encoders, onehot_encoded_cols = CategoricalEncoder.fit_transform_onehot_encoding(cleaned_data, categorical_columns, max_categories=max_categories)
    print("\nDataFrame after One-Hot Encoding:")
    print(cleaned_data.head())
    print("\nOne-Hot Encoded Columns:")
    print(onehot_encoded_cols)

# Transform the data
transformer = DataTransformer()
numeric_columns = cleaned_data.select_dtypes(include=[np.number]).columns.tolist()
if label in numeric_columns:
    numeric_columns.remove(label)

# Apply log transformation to numeric columns to handle skewness
for column in numeric_columns:
    skewness = cleaned_data[column].skew()
    if abs(skewness) > 0.5:
        cleaned_data = transformer.apply_log_transformation(cleaned_data, column)
        print(f"Applied log transformation to {column} due to high skewness ({skewness}).")

# Detect and handle outliers
for column in numeric_columns:
    Q1 = cleaned_data[column].quantile(0.25)
    Q3 = cleaned_data[column].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    outliers = ((cleaned_data[column] < lower_bound) | (cleaned_data[column] > upper_bound)).sum()
    if outliers > 0:
        cleaned_data = transformer.impute_outliers(cleaned_data, column, method='median')
        print(f"Imputed outliers in {column} ({outliers} outliers found).")

    cleaned_data = transformer.winsorize_data(cleaned_data, column, limits=(0.05, 0.05))
    print(f"Applied Winsorizing to {column}.")

transformed_df = cleaned_data.copy()

print("\nTransformed Data:")
print(transformed_df.head())

# Feature engineering
feature_engineer = FeatureEngineer()

# Automatically evaluate and add features
transformed_df = feature_engineer.evaluate_and_add_all_features(transformed_df, label)

print("\nDataFrame after Feature Engineering:")
print(transformed_df.head())

# Scale the data
scaler = SimpleScaler()
numeric_cols = transformed_df.select_dtypes(include=[np.number]).columns.tolist()
if label in numeric_cols:
    numeric_cols.remove(label)

standardized_cols = {}
normalized_cols = {}

# Standardize and normalize all numeric columns excluding OHE columns
for col in numeric_cols:
    if col not in onehot_encoded_cols:  # Ensure the column is not an OHE column
        standardized_data, _, _ = scaler.standardize(transformed_df[col].values)
        normalized_data, _, _ = scaler.normalize(transformed_df[col].values)
        standardized_cols[f'Standardized_{col}'] = standardized_data
        normalized_cols[f'Normalized_{col}'] = normalized_data

# Add standardized and normalized columns back to the DataFrame
for col_name, col_data in standardized_cols.items():
    transformed_df[col_name] = col_data

for col_name, col_data in normalized_cols.items():
    transformed_df[col_name] = col_data

print("\nDataFrame with Standardized and Normalized Columns:")
print(transformed_df.head())

# Optional: Save the cleaned, transformed, and scaled DataFrame to a new CSV file
transformed_df.to_csv('movies_scaled.csv', index=False)
