import pandas as pd
import numpy as np
from scipy.stats.mstats import winsorize

class DataTransformer:

    @staticmethod
    def apply_log_transformation(dataframe, column):
        """Applies log transformation to the specified column to reduce skewness."""
        transformed_data = dataframe.copy()
        transformed_data[column] = np.log(transformed_data[column] + 1)
        return transformed_data

    @staticmethod
    def reverse_log_transformation(transformed_dataframe, column):
        """Reverses the log transformation if needed."""
        reversed_data = transformed_dataframe.copy()
        reversed_data[column] = np.exp(reversed_data[column]) - 1
        return reversed_data

    @staticmethod
    def impute_outliers(dataframe, column, method='median'):
        """Imputes outliers in the specified column using the median or mean."""
        df = dataframe.copy()
        Q1 = df[column].quantile(0.25)
        Q3 = df[column].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        
        if method == 'median':
            replacement_value = df[column].median()
        elif method == 'mean':
            replacement_value = df[column].mean()
        else:
            raise ValueError("Method must be 'median' or 'mean'")

        df.loc[df[column] < lower_bound, column] = replacement_value
        df.loc[df[column] > upper_bound, column] = replacement_value
        return df
    
    @staticmethod
    def apply_log_transformation(dataframe, column):
        """Applies log transformation to reduce skewness."""
        transformed_data = dataframe.copy()
        transformed_data[column] = np.log(transformed_data[column] + 1)
        return transformed_data

    @staticmethod
    def reverse_log_transformation(transformed_dataframe, column):
        """Reverses the log transformation."""
        reversed_data = transformed_dataframe.copy()
        reversed_data[column] = np.exp(reversed_data[column]) - 1
        return reversed_data

    @staticmethod
    def winsorize_data(dataframe, column, limits):
        """Applies Winsorizing to the specified column to limit extreme values."""
        df = dataframe.copy()
        df[column] = winsorize(df[column], limits=limits)
        return df

