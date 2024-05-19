import numpy as np

class dataCleaner:
    def remove_rows_with_missing_labels(self, dataset, label_column):
        """
        Removes rows from the DataFrame where the label column has NaN values.
        """
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
        numeric_cols = dataset.select_dtypes(include=np.number).columns.tolist()
        means = dataset[numeric_cols].mean()  # Calculate the means here
        
        for column in dataset.columns:
            if column != label_column:
                missing_data_ratio = dataset[column].isnull().sum() / len(dataset)
                
                # Handle numeric columns
                if column in numeric_cols:
                    if missing_data_ratio < missing_threshold:
                        dataset[column] = dataset[column].fillna(means[column])
                    elif missing_data_ratio >= missing_threshold:
                        correlation_with_label = dataset[numeric_cols + [label_column]].corr().loc[column, label_column]
                        
                        # Drop column if correlation is below threshold
                        if abs(correlation_with_label) < correlation_threshold:
                            dataset.drop(columns=[column], inplace=True)
                # Handle non-numeric columns
                else:
                    if missing_data_ratio >= missing_threshold:
                        dataset.drop(columns=[column], inplace=True)
        return dataset
