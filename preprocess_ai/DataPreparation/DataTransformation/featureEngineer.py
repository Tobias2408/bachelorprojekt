import pandas as pd
import numpy as np

class FeatureEngineer:
    @staticmethod
    def add_interaction_feature(df, feature1, feature2, new_feature_name):
        if df.empty or feature1 not in df.columns or feature2 not in df.columns:
            return df
        df[new_feature_name] = df[feature1] * df[feature2]
        return df

    @staticmethod
    def add_date_parts(df, date_column):
        if df.empty or date_column not in df.columns:
            return df
        df[date_column] = pd.to_datetime(df[date_column])
        df[f'{date_column}_year'] = df[date_column].dt.year
        df[f'{date_column}_month'] = df[date_column].dt.month
        df[f'{date_column}_day'] = df[date_column].dt.day
        df[f'{date_column}_dayofweek'] = df[date_column].dt.dayofweek
        df[f'{date_column}_is_weekend'] = df[date_column].dt.dayofweek.isin([5, 6]).astype('Int32')
        return df

    @staticmethod
    def add_rolling_features(df, column, window_size):
        if df.empty or column not in df.columns:
            return df
        df[f'{column}_rolling_mean_{window_size}'] = df[column].rolling(window=window_size).mean()
        df[f'{column}_rolling_std_{window_size}'] = df[column].rolling(window=window_size).std()
        return df
