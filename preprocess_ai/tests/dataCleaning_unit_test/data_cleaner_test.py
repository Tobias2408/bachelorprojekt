import unittest
import pandas as pd
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from preprocess_ai.DataPreparation.DataCleaner.dataCleaner import DataCleaner

class DataCleanerTest(unittest.TestCase):
    def setUp(self):
        """Set up a new DataCleaner and sample DataFrame for each test."""
        self.cleaner = DataCleaner()

    def get_sample_df(self):
        """Create a sample DataFrame to use in tests with known correlations."""
        data = {
            'Feature1': [1, 2, 3, 4, 5, 6, None, 8, 9, 10],  # Linear correlation with Label
            'Feature2': [2, 4, 6, 8, None, 12, 14, 16, 18, 20],  # Linear correlation with Label
            'Feature3': [1, None, 1, 2, 1, 2, 1, None, 2, 1],  # No clear correlation
            'Label': [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]  # Label
        }
        return pd.DataFrame(data)

    def test_remove_nan_rows(self):
        """Test that rows with NaN values are removed correctly."""
        df = self.get_sample_df()
        cleaned_df = self.cleaner.remove_NaN_rows(df)
        self.assertEqual(len(cleaned_df), 6)  # Only 6 rows should have no NaNs

    def test_smart_clean_data_fill_missing_with_mean(self):
        """Test that missing numeric values are filled with the mean after removing rows with missing labels."""
        df = self.get_sample_df()
        cleaned_df = self.cleaner.smart_clean_data(df, 'Label', 0.3, 0.5)
        expected_mean_feature1 = (1 + 2 + 3 + 4 + 5 + 6 + 8 + 9 + 10) / 9  # Mean of remaining values in Feature1
        expected_mean_feature2 = (2 + 4 + 6 + 8 + 12 + 14 + 16 + 18 + 20) / 9  # Mean of remaining values in Feature2
        self.assertAlmostEqual(cleaned_df['Feature1'].iloc[6], expected_mean_feature1, places=7)
        self.assertAlmostEqual(cleaned_df['Feature2'].iloc[4], expected_mean_feature2, places=7)
        self.assertFalse(cleaned_df.isnull().values.any(), "There should be no NaN values in the cleaned DataFrame")

    def test_smart_clean_data_retain_non_numeric_column(self):
        """Test that non-numeric columns with missing data are not removed."""
        df = self.get_sample_df()
        cleaned_df = self.cleaner.smart_clean_data(df, 'Label', 0.3, 0.5)
        self.assertIn('Feature3', cleaned_df.columns)  # Verify 'Feature3' is not removed
        self.assertFalse(cleaned_df.isnull().values.any(), "There should be no NaN values in the cleaned DataFrame")

    def test_smart_clean_data_fill_missing_with_mean_specific_threshold(self):
        """Test that missing numeric values are filled with the mean."""
        df = self.get_sample_df()
        cleaned_df = self.cleaner.smart_clean_data(df, 'Label', 0.4, 0.5)
        expected_mean_feature1 = (1 + 2 + 3 + 4 + 5 + 6 + 8 + 9 + 10) / 9  # Calculate expected mean after dropping rows with missing labels
        expected_mean_feature2 = (2 + 4 + 6 + 8 + 12 + 14 + 16 + 18 + 20) / 9  # Mean of remaining values in Feature2
        self.assertAlmostEqual(cleaned_df['Feature1'].iloc[6], expected_mean_feature1, places=7)
        self.assertAlmostEqual(cleaned_df['Feature2'].iloc[4], expected_mean_feature2, places=7)
        self.assertFalse(cleaned_df.isnull().values.any(), "There should be no NaN values in the cleaned DataFrame")

    def test_smart_clean_data_retain_columns(self):
        """Test that columns with low missing data ratios are not removed."""
        df = self.get_sample_df()
        cleaned_df = self.cleaner.smart_clean_data(df, 'Label', 0.3, 0.5)
        self.assertIn('Feature2', cleaned_df.columns)
        self.assertFalse(cleaned_df.isnull().values.any(), "There should be no NaN values in the cleaned DataFrame")

if __name__ == '__main__':
    unittest.main()
