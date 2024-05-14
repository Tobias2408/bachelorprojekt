import pandas as pd
import numpy as np

class FeatureEngineer:
    
    @staticmethod
    def add_custom_feature(df, feature_name, func, *args, **kwargs):
        """
        Adds a custom feature to the DataFrame by applying the specified function.
        
        Parameters:
        df (pd.DataFrame): DataFrame containing the data.
        feature_name (str): Name of the new feature.
        func (callable): Function to apply to create the new feature.
        *args: Positional arguments to pass to the function.
        **kwargs: Keyword arguments to pass to the function.
        
        Returns:
        pd.DataFrame: DataFrame with the new feature.
        """
        df[feature_name] = func(df, *args, **kwargs)
        return df

# Example custom functions

def interaction_feature(df, feature1, feature2):
    return df[feature1] * df[feature2]

def date_parts(df, date_column):
    df[date_column] = pd.to_datetime(df[date_column])
    return pd.concat([
        df[date_column].dt.year.rename(f"{date_column}_year"),
        df[date_column].dt.month.rename(f"{date_column}_month"),
        df[date_column].dt.day.rename(f"{date_column}_day"),
        df[date_column].dt.dayofweek.rename(f"{date_column}_dayofweek"),
        (df[date_column].dt.dayofweek >= 5).rename(f"{date_column}_is_weekend").astype(int)
    ], axis=1)

def rolling_features(df, feature, window_size):
    return pd.concat([
        df[feature].rolling(window=window_size).mean().rename(f"{feature}_rolling_mean_{window_size}"),
        df[feature].rolling(window=window_size).std().rename(f"{feature}_rolling_std_{window_size}")
    ], axis=1)

def total_sales(df, price_col, quantity_col):
    return df[price_col] * df[quantity_col]

