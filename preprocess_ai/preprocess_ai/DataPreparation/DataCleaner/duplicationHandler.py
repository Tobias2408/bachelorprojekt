import pandas as pd

class DuplicationHandler:
    @staticmethod
    def identify_duplicates(dataframe, subset=None):
        """
        Identify duplicate rows in the DataFrame.
        
        Parameters:
        dataframe (pd.DataFrame): The DataFrame to handle.
        subset (list of str, optional): List of columns to consider for identifying duplicates.
        
        Returns:
        pd.DataFrame: DataFrame containing only the duplicate rows.
        """
        duplicates = dataframe.duplicated(subset=subset, keep=False)
        return dataframe[duplicates]

    @staticmethod
    def remove_duplicates(dataframe, subset=None):
        """
        Remove duplicate rows from the DataFrame, keeping the first occurrence.
        
        Parameters:
        dataframe (pd.DataFrame): The DataFrame to handle.
        subset (list of str, optional): List of columns to consider for identifying duplicates.
        
        Returns:
        pd.DataFrame: DataFrame after duplicates have been removed.
        """
        return dataframe.drop_duplicates()


    @staticmethod
    def log_duplicates(dataframe, subset=None, filepath='duplicates_log.csv'):
        """
        Log the duplicate rows into a CSV file and return these rows.
        
        Parameters:
        dataframe (pd.DataFrame): The DataFrame to handle.
        subset (list of str, optional): List of columns to consider for identifying duplicates.
        filepath (str): File path to save the log file.
        
        Returns:
        pd.DataFrame: DataFrame of duplicate rows that were logged.
        """
        duplicates = dataframe.duplicated(subset=subset, keep=False)
        duplicate_df = dataframe[duplicates]
        duplicate_df.to_csv(filepath, index=False)
        return duplicate_df

    @staticmethod
    def aggregate_duplicates(dataframe, key_columns, agg_column, agg_func='mean'):
        """
        Aggregate duplicate rows based on the specified key columns and aggregation function.
        
        Parameters:
        dataframe (pd.DataFrame): The DataFrame to handle.
        key_columns (list of str): Columns to group by.
        agg_column (str): Column to aggregate.
        agg_func (str): Aggregation function to apply (e.g., 'mean', 'sum', 'max').
        
        Returns:
        pd.DataFrame: A new DataFrame with duplicates aggregated.
        """
        grouped = dataframe.groupby(key_columns)
        if agg_func == 'mean':
            result = grouped[agg_column].mean().reset_index()
        elif agg_func == 'sum':
            result = grouped[agg_column].sum().reset_index()
        elif agg_func == 'max':
            result = grouped[agg_column].max().reset_index()
        else:
            raise ValueError("Unsupported aggregation function.")
        return result
