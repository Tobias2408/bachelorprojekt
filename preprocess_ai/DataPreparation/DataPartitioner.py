import pandas as pd
from sklearn.model_selection import train_test_split

class DataPartitioner:
    
    @staticmethod
    def split_data(df, target, test_size=0.2, val_size=0.1, stratify=False):
        """
        Splits the data into training, validation, and test sets.
        
        Parameters:
        df (pd.DataFrame): DataFrame containing the data.
        target (str): Name of the target variable.
        test_size (float): Proportion of the data to include in the test set.
        val_size (float): Proportion of the data to include in the validation set.
        stratify (bool): Whether to use stratified sampling based on the target variable.
        
        Returns:
        tuple: DataFrames for training, validation, and test sets (X_train, X_val, X_test, y_train, y_val, y_test).
        """
        stratify_col = df[target] if stratify else None
        
        # Split into train+val and test sets
        X_train_val, X_test, y_train_val, y_test = train_test_split(
            df.drop(columns=[target]), df[target], test_size=test_size, stratify=stratify_col, random_state=42
        )
        
        # Ensure val_size is at least as large as the number of classes divided by the total number of samples
        min_val_size = len(df[target].unique()) / len(df)
        if val_size < min_val_size:
            raise ValueError(f"val_size should be at least {min_val_size} to ensure each class is represented in the validation set.")
        
        # Calculate validation size as a proportion of the training+validation set
        val_size_adjusted = val_size / (1 - test_size)
        
        # Split train+val set into train and validation sets
        stratify_col_train_val = y_train_val if stratify else None
        X_train, X_val, y_train, y_val = train_test_split(
            X_train_val, y_train_val, test_size=val_size_adjusted, stratify=stratify_col_train_val, random_state=42
        )
        
        return X_train, X_val, X_test, y_train, y_val, y_test
