import numpy as np
class dataCleaner:
    def remove_rows_with_missing_labels(self, dataset, label_column):
        """
        Removes rows from the DataFrame where the label column has NaN values.

        Parameters:
        dataset (pd.DataFrame): The DataFrame to clean.
        label_column (str): The name of the label column.

        Returns:
        pd.DataFrame: A DataFrame with rows removed where the label column has NaN values.
        """
        # Drop rows where the label column is NaN
        cleaned_dataset = dataset.dropna(subset=[label_column])
        return cleaned_dataset

    def remove_NaN_rows(self, dataset):
        """
        Cleans the DataFrame by removing rows that contain NaN values.
        """
        if dataset.isnull().values.any():
            cleaned_dataset = dataset.dropna()
            return cleaned_dataset
        else:
            return dataset

    def clean_data_columns(self, dataset):
        """
        Cleans the DataFrame by removing columns that contain NaN values.
        """
        if dataset.isnull().values.any():
            cleaned_dataset = dataset.dropna(axis=1)
            return cleaned_dataset
        else:
            return dataset

    def fill_missing_with_mean(self, dataset):
        """
        Fills missing values in numeric columns with the mean of each column.
        """
        filled_dataset = dataset.copy()
        for column in filled_dataset.columns:
            if filled_dataset[column].dtype in ['float64', 'int64']:
                mean_value = filled_dataset[column].mean()
                filled_dataset[column] = filled_dataset[column].fillna(mean_value)
        return filled_dataset

    def smart_clean_data(self, dataset, label_column, missing_threshold=0.3, correlation_threshold=0.5):
        """
        Cleans the dataset intelligently:
        1. Removes rows where the label column has NaN values.
        2. Fills missing values for columns with a low ratio of missing data, ensuring numeric type.
        3. Drops columns with a high missing data ratio and low correlation with the label column if they are numeric.
        """
        # Remove rows with missing label data and operate on a copy to avoid SettingWithCopyWarning
        dataset = self.remove_rows_with_missing_labels(dataset.copy(), label_column)

        # Re-calculate means for numeric columns after removing rows
        numeric_cols = dataset.select_dtypes(include=[np.number]).columns
        means = {col: dataset[col].mean(skipna=True) for col in numeric_cols if col != label_column}

        # Process all columns, fill numeric with means, handle non-numeric differently if needed
        for column in dataset.columns:
            if column != label_column:
                missing_data_ratio = dataset[column].isnull().sum() / len(dataset)

                # Handle numeric columns
                if column in numeric_cols:
                    if missing_data_ratio < missing_threshold:
                        dataset[column] = dataset[column].fillna(means[column])
                    elif missing_data_ratio >= missing_threshold:
                        correlation_with_label = dataset[numeric_cols].corr().loc[column, label_column]
                        if abs(correlation_with_label) < correlation_threshold:
                            dataset.drop(column, axis=1, inplace=True)
                else:
                    # Optional: Handle non-numeric columns, e.g., filling with mode or a constant
                    if missing_data_ratio > 0:
                        # Fill with the most common value (mode) or another appropriate value
                        mode_value = dataset[column].mode().iloc[0] if not dataset[column].mode().empty else 'Unknown'
                        dataset[column] = dataset[column].fillna(mode_value)

        return dataset


