import unittest
import pandas as pd
import numpy as np
from scipy.stats import mstats

from DataPreparation.DataCleaner.dataTransformer import DataTransformer  # Assuming DataTransformer is in 'dataTransformer.py'

class TestDataCleanerTransformer(unittest.TestCase):

    def setUp(self):
        """Set up test variables for all tests."""
        self.data = pd.DataFrame({
            'Values': [1, 2, 2, 3, 3, 3, 4, 4, 4, 4, 10, 20, 50, 100]
        })

    def test_apply_log_transformation(self):
        """Test log transformation on 'Values' column."""
        transformed_df = DataTransformer.apply_log_transformation(self.data, 'Values')
        expected = np.log(self.data['Values'] + 1)
        pd.testing.assert_series_equal(transformed_df['Values'], expected)

    def test_reverse_log_transformation(self):
        """Test reversing the log transformation on 'Values' column."""
        transformed_df = DataTransformer.apply_log_transformation(self.data, 'Values')
        reversed_df = DataTransformer.reverse_log_transformation(transformed_df, 'Values')
        is_close = np.isclose(reversed_df['Values'].values, self.data['Values'].values, atol=1e-6)
        self.assertTrue(np.all(is_close), "Not all values are close within the tolerance after reversing transformation.")

    def test_impute_outliers_with_median(self):
        """Test outlier imputation using median."""
        median_before = self.data['Values'].median()
        imputed_df = DataTransformer.impute_outliers(self.data, 'Values', method='median')
        # Check that the previous outliers are now equal to the median
        self.assertTrue((imputed_df['Values'] == median_before).all(), "Outliers were not all replaced by the median")

    def test_impute_outliers_with_median(self):
        """Test outlier imputation using median."""
        df_copy = self.data.copy()
        median_before = df_copy['Values'].median()

        # Calculate IQR to identify outliers manually
        Q1 = df_copy['Values'].quantile(0.25)
        Q3 = df_copy['Values'].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR

        # Apply imputation
        imputed_df = DataTransformer.impute_outliers(df_copy, 'Values', method='median')
        
        # Check only outliers are replaced
        outliers = df_copy[(df_copy['Values'] < lower_bound) | (df_copy['Values'] > upper_bound)]
        non_outliers = df_copy[~((df_copy['Values'] < lower_bound) | (df_copy['Values'] > upper_bound))]

        # Verify outliers are replaced by the median
        for index in outliers.index:
            self.assertEqual(imputed_df.loc[index, 'Values'], median_before, "Outlier was not replaced by median correctly.")

        # Verify non-outliers are unchanged
        pd.testing.assert_series_equal(imputed_df.loc[non_outliers.index, 'Values'], non_outliers['Values'], check_names=False)

    def test_impute_outliers_invalid_method(self):
        """Test passing an invalid method to impute_outliers."""
        with self.assertRaises(ValueError):
            DataTransformer.impute_outliers(self.data, 'Values', method='invalid')
    
    def test_winsorize_data(self):
        """Test Winsorizing on 'Values' column."""
        # Applying Winsorizing with limits for both tails
        winsorized_df = DataTransformer.winsorize_data(self.data, 'Values', limits=(0.1, 0.1))
        
        # Expected behavior: The top 10% and bottom 10% should be replaced by the 10th and 90th percentiles respectively
        expected_winsorized_values = mstats.winsorize(self.data['Values'], limits=(0.1, 0.1))
        
        # Check that Winsorized values match expected values
        pd.testing.assert_series_equal(winsorized_df['Values'], pd.Series(expected_winsorized_values), check_names=False)

    def test_winsorize_data_extreme_limits(self):
        """Test Winsorizing with extreme limits."""
        # Applying Winsorizing with limits that cap more extreme values
        winsorized_df = DataTransformer.winsorize_data(self.data, 'Values', limits=(0.2, 0.2))
        
        # Expected behavior: The top 20% and bottom 20% should be replaced
        expected_winsorized_values = mstats.winsorize(self.data['Values'], limits=(0.2, 0.2))
        
        # Check that Winsorized values match expected values
        pd.testing.assert_series_equal(winsorized_df['Values'], pd.Series(expected_winsorized_values), check_names=False)

if __name__ == '__main__':
    unittest.main()
