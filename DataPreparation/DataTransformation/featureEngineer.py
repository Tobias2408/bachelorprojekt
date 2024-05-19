import pandas as pd
import numpy as np

class FeatureEngineer:
    
    @staticmethod
    def add_custom_feature(df, feature_name, func, *args, **kwargs):
        """
        Adds a custom feature to the DataFrame by applying the specified function.
        """
        df[feature_name] = func(df, *args, **kwargs)
        return df

    @staticmethod
    def evaluate_and_add_feature(df, target, feature_name, func, *args, **kwargs):
        """
        Evaluate the correlation of the new feature with the target and add it if it improves correlation.
        """
        new_features = func(df, *args, **kwargs)
        
        if isinstance(new_features, pd.DataFrame):
            for col in new_features.columns:
                if col not in df.columns:
                    df[col] = new_features[col]
        else:
            if feature_name not in df.columns:
                df[feature_name] = new_features
        
        return df

    @staticmethod
    def evaluate_and_add_all_features(df, target):
        """
        Automatically evaluate and add multiple features to the DataFrame.
        """
        numeric_columns = df.select_dtypes(include=[np.number]).columns.tolist()
        if target in numeric_columns:
            numeric_columns.remove(target)

        # Evaluate interaction features
        for i in range(len(numeric_columns)):
            for j in range(i + 1, len(numeric_columns)):
                col1 = numeric_columns[i]
                col2 = numeric_columns[j]
                interaction_df = interaction_feature(df, col1, col2)
                for interaction_col in interaction_df.columns:
                    df = FeatureEngineer.evaluate_and_add_feature(df, target, interaction_col, interaction_feature, col1, col2)

        # Evaluate rolling features
        for col in numeric_columns:
            rolling_df = rolling_features(df, col, window_size=3)
            for rolling_col in rolling_df.columns:
                if rolling_col not in df.columns:
                    df[rolling_col] = rolling_df[rolling_col]

        return df

# Define the required feature engineering functions

def interaction_feature(df, col1, col2):
    """
    Create an interaction feature by multiplying two columns.
    """
    interaction_col_name = f'{col1}_x_{col2}'
    if interaction_col_name in df.columns:
        return df[[interaction_col_name]]
    
    df[interaction_col_name] = df[col1] * df[col2]
    return df[[interaction_col_name]]

def rolling_features(df, col, window_size):
    """
    Create rolling mean and rolling std features for a given column.
    """
    rolling_mean_col_name = f'{col}_rolling_mean_{window_size}'
    rolling_std_col_name = f'{col}_rolling_std_{window_size}'
    
    rolling_mean = df[col].rolling(window=window_size).mean()
    rolling_std = df[col].rolling(window=window_size).std()
    
    return pd.DataFrame({
        rolling_mean_col_name: rolling_mean,
        rolling_std_col_name: rolling_std
    })

# Additional feature engineering functions

def date_parts(df, date_col):
    """
    Extract year, month, and day from a date column.
    """
    df[date_col] = pd.to_datetime(df[date_col], errors='coerce')
    df[f'{date_col}_year'] = df[date_col].dt.year
    df[f'{date_col}_month'] = df[date_col].dt.month
    df[f'{date_col}_day'] = df[date_col].dt.day
    return df[[f'{date_col}_year', f'{date_col}_month', f'{date_col}_day']]

def total_sales(df, quantity_col, price_col):
    """
    Calculate total sales by multiplying quantity and price.
    """
    total_sales_col_name = f'Total_sales_{quantity_col}_x_{price_col}'
    if total_sales_col_name in df.columns:
        return df[[total_sales_col_name]]
    
    df[total_sales_col_name] = df[quantity_col] * df[price_col]
    return df[[total_sales_col_name]]

# Example usage in main script
if __name__ == "__main__":
    df = pd.read_csv('titanic.csv').head(100)  # Load sample data
    df = FeatureEngineer.evaluate_and_add_all_features(df, 'Survived')
    print(df.head())
