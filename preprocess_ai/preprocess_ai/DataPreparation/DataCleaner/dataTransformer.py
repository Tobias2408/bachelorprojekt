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
    @staticmethod
    def auto_transform(df, target, outlier_method='median', log_threshold=0.5, outlier_threshold=1.5, winsorize_limits=(0.05, 0.05)):
        """
        Automatically applies transformations based on data characteristics.
        
        Parameters:
        df (pd.DataFrame): DataFrame containing the data.
        target (str): The target column name.
        outlier_method (str): Method to use for imputing outliers ('median' or 'mean').
        log_threshold (float): Skewness threshold above which log transformation is applied.
        outlier_threshold (float): IQR multiplier threshold for detecting outliers.
        winsorize_limits (tuple): Limits for Winsorizing the data.
        
        Returns:
        pd.DataFrame: Transformed DataFrame.
        """
        transformed_df = df.copy()
        numeric_columns = transformed_df.select_dtypes(include=[np.number]).columns.tolist()
        numeric_columns.remove(target)
        
        for column in numeric_columns:
            # Check skewness and apply log transformation if necessary
            skewness = transformed_df[column].skew()
            if abs(skewness) > log_threshold:
                transformed_df = DataTransformer.apply_log_transformation(transformed_df, column)
                print(f"Applied log transformation to {column} due to high skewness ({skewness}).")
            
            # Check for outliers and apply imputation
            Q1 = transformed_df[column].quantile(0.25)
            Q3 = transformed_df[column].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - outlier_threshold * IQR
            upper_bound = Q3 + outlier_threshold * IQR
            outliers = ((transformed_df[column] < lower_bound) | (transformed_df[column] > upper_bound)).sum()
            if outliers > 0:
                transformed_df = DataTransformer.impute_outliers(transformed_df, column, method=outlier_method)
                print(f"Imputed outliers in {column} ({outliers} outliers found).")
            
            # Apply Winsorizing
            transformed_df[column] = winsorize(transformed_df[column], limits=winsorize_limits)
            print(f"Applied Winsorizing to {column}.")

        return transformed_df
