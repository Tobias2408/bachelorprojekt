import numpy as np
import pandas as pd

class DataCleaner:
    def remove_rows_with_missing_labels(self, dataset, label_column):
        """
        Removes rows from the DataFrame where the label column has NaN values.
        """
        return dataset.dropna(subset=[label_column])

    def remove_NaN_rows(self, dataset):
        """
        Cleans the DataFrame by removing rows that contain NaN values.
        """
        return dataset.dropna()

    def clean_data_columns(self, dataset):
        """
        Cleans the DataFrame by removing columns that contain NaN values.
        """
        return dataset.dropna(axis=1)

    def fill_missing_with_mean(self, dataset):
        """
        Fills missing values in numeric columns with the mean of each column.
        """
        filled_dataset = dataset.copy()
        for column in filled_dataset.columns:
            if pd.api.types.is_numeric_dtype(filled_dataset[column]):
                mean_value = filled_dataset[column].mean()
                filled_dataset[column] = filled_dataset[column].fillna(mean_value)
        return filled_dataset

    def smart_clean_data(self, dataset, label_column, missing_threshold=0.3, correlation_threshold=0.5):
        """
        Cleans the DataFrame by handling missing data and removing columns based on correlation with the label.
        """
        # Step 1: Remove rows with missing labels
        dataset = self.remove_rows_with_missing_labels(dataset, label_column).copy()

        # Precompute means for numeric columns only
        numeric_cols = dataset.select_dtypes(include=np.number).columns
        means = dataset[numeric_cols].mean()

        # Step 2: Fill missing numeric values with the precomputed means
        for column in numeric_cols:
            dataset[column] = dataset[column].fillna(means[column])

        # Calculate correlation matrix including the label if it's numeric
        if label_column in numeric_cols:
            numeric_cols_with_label = numeric_cols.tolist()
        else:
            numeric_cols_with_label = numeric_cols.tolist() + [label_column]

        correlation_matrix = dataset[numeric_cols_with_label].corr()

        columns_to_drop = []
        for column in dataset.columns:
            if column != label_column:
                missing_data_ratio = dataset[column].isnull().sum() / len(dataset)

                if column in numeric_cols:
                    if missing_data_ratio >= missing_threshold:
                        if column in correlation_matrix.index and label_column in correlation_matrix.columns:
                            correlation_with_label = correlation_matrix.loc[column, label_column]
                            if np.abs(correlation_with_label) < correlation_threshold:
                                columns_to_drop.append(column)
                else:
                    if missing_data_ratio >= missing_threshold:
                        columns_to_drop.append(column)

        # Drop columns outside the loop to avoid SettingWithCopyWarning
        dataset = dataset.drop(columns=columns_to_drop).copy()

        # Ensure no NaN values are left in numeric columns
        for column in numeric_cols:
            if column in dataset.columns:
                dataset[column] = dataset[column].fillna(means[column])

        return dataset
