import unittest
import pandas as pd
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from DataPreparation.DataCleaner.dataCleaner import dataCleaner

class data_cleaner_test(unittest.TestCase):
    def setUp(self):
        """Create a sample DataFrame to use in tests."""
        self.data = {
            'Feature1': [1, 2, None, 4],  # Some missing, should be filled
            'Feature2': [4, None, 2, 1],  # Some missing, but not enough to drop
            'Feature3': ['A', None, 'B', 'C'],  # Non-numeric, should remain unaffected
            'Label': [10, 11, 12, None]  # Some missing in the label
        }
        self.df = pd.DataFrame(self.data)
        self.cleaner = dataCleaner()

    def test_remove_nan_rows(self):
        """Test that rows with NaN values are removed correctly."""
        cleaned_df = self.cleaner.remove_NaN_rows(self.df)
        self.assertEqual(len(cleaned_df), 1)  # Only 1 row should have no NaNs

    def test_smart_clean_data_fill_missing_with_mean(self):
        """Test that missing numeric values are filled with the mean after removing rows with missing labels."""
        cleaned_df = self.cleaner.smart_clean_data(self.df, 'Label', 0.3, 0.5)
        expected_mean = (1 + 2 +4) / 2  # Updated expected mean after dropping row with missing label
        self.assertEqual(cleaned_df['Feature1'].iloc[1], expected_mean, "Feature1 should be filled with the correct mean")
        print(cleaned_df)

    def test_smart_clean_data_retain_non_numeric_column(self):
        """Test that non-numeric columns with missing data are not removed."""
        cleaned_df = self.cleaner.smart_clean_data(self.df, 'Label', 0.3, 0.5)
        self.assertIn('Feature3', cleaned_df.columns)  # Verify 'Feature3' is not removed

    def test_smart_clean_data_fill_missing_with_mean(self):
        """Test that missing numeric values are filled with the mean."""
        cleaned_df = self.cleaner.smart_clean_data(self.df, 'Label', 0.4, 0.5)
        print(cleaned_df)
        expected_mean = (1 + 2 ) / 2  # Calculate expected mean
        self.assertEqual(cleaned_df['Feature1'].iloc[2], expected_mean)

    def test_smart_clean_data_retain_columns(self):
        """Test that columns with low missing data ratios are not removed."""
        cleaned_df = self.cleaner.smart_clean_data(self.df, 'Label', 0.3, 0.5)
        self.assertTrue('Feature2' in cleaned_df.columns)

if __name__ == '__main__':
    unittest.main()
