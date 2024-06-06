import pandas as pd

class OutlierHandler:
    def __init__(self, dataframe, column):
        self.df = dataframe
        self.column = column

    def calculate_iqr(self):
        """Calculate the interquartile range of the column."""
        Q1 = self.df[self.column].quantile(0.25)
        Q3 = self.df[self.column].quantile(0.75)
        IQR = Q3 - Q1
        return Q1, Q3, IQR

    def find_outliers(self):
        """Identify outliers in the dataframe based on the IQR method."""
        Q1, Q3, IQR = self.calculate_iqr()
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        
        outliers = self.df[(self.df[self.column] < lower_bound) | (self.df[self.column] > upper_bound)]
        return outliers
    
    def remove_outliers(self):
        """Remove outliers from the dataframe."""
        Q1, Q3, IQR = self.calculate_iqr()
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        
        # Keep rows that are not outliers
        self.df = self.df[(self.df[self.column] >= lower_bound) & (self.df[self.column] <= upper_bound)]
        return self.df