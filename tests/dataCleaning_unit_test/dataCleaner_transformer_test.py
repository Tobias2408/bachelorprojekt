import unittest
import pandas as pd
import numpy as np
from scipy.stats.mstats import winsorize
from DataPreparation.DataCleaner.dataTransformer import DataTransformer  # Assuming DataTransformer is in 'dataTransformer.py'

class TestDataCleanerTransformer(unittest.TestCase):

    def setUp(self):
        """Set up test variables for all tests."""
        self.data = pd.DataFrame({
            'feature1': [1, 2, 2, 3, 3, 3, 4, 4, 4, 4, 10, 20, 50, 100],
            'feature2': [10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 110, 120, 130, 140],
            'target': [1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0]
        })

    def test_apply_log_transformation(self):
        """Test log transformation on 'feature1' column."""
        transformed_df = DataTransformer.apply_log_transformation(self.data, 'feature1')
        expected = np.log(self.data['feature1'] + 1)
        pd.testing.assert_series_equal(transformed_df['feature1'], expected)

    def test_reverse_log_transformation(self):
        """Test reversing the log transformation on 'feature1' column."""
        transformed_df = DataTransformer.apply_log_transformation(self.data, 'feature1')
        reversed_df = DataTransformer.reverse_log_transformation(transformed_df, 'feature1')
        is_close = np.isclose(reversed_df['feature1'].values, self.data['feature1'].values, atol=1e-6)
        self.assertTrue(np.all(is_close), "Not all values are close within the tolerance after reversing transformation.")

    def test_impute_outliers_with_median(self):
        """Test outlier imputation using median."""
        df_copy = self.data.copy()
        median_before = df_copy['feature1'].median()

        # Calculate IQR to identify outliers manually
        Q1 = df_copy['feature1'].quantile(0.25)
        Q3 = df_copy['feature1'].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR

        # Apply imputation
        imputed_df = DataTransformer.impute_outliers(df_copy, 'feature1', method='median')
        
        # Check only outliers are replaced
        outliers = df_copy[(df_copy['feature1'] < lower_bound) | (df_copy['feature1'] > upper_bound)]
        non_outliers = df_copy[~((df_copy['feature1'] < lower_bound) | (df_copy['feature1'] > upper_bound))]

        # Verify outliers are replaced by the median
        for index in outliers.index:
            self.assertEqual(imputed_df.loc[index, 'feature1'], median_before, "Outlier was not replaced by median correctly.")

        # Verify non-outliers are unchanged
        pd.testing.assert_series_equal(imputed_df.loc[non_outliers.index, 'feature1'], non_outliers['feature1'], check_names=False)

    def test_impute_outliers_invalid_method(self):
        """Test passing an invalid method to impute_outliers."""
        with self.assertRaises(ValueError):
            DataTransformer.impute_outliers(self.data, 'feature1', method='invalid')
    
    def test_winsorize_data(self):
        """Test Winsorizing on 'feature1' column."""
        # Applying Winsorizing with limits for both tails
        winsorized_df = DataTransformer.winsorize_data(self.data, 'feature1', limits=(0.1, 0.1))
        
        # Expected behavior: The top 10% and bottom 10% should be replaced by the 10th and 90th percentiles respectively
        expected_winsorized_values = winsorize(self.data['feature1'], limits=(0.1, 0.1))
        
        # Check that Winsorized values match expected values
        pd.testing.assert_series_equal(winsorized_df['feature1'], pd.Series(expected_winsorized_values), check_names=False)

    def test_winsorize_data_extreme_limits(self):
        """Test Winsorizing with extreme limits."""
        # Applying Winsorizing with limits that cap more extreme values
        winsorized_df = DataTransformer.winsorize_data(self.data, 'feature1', limits=(0.2, 0.2))
        
        # Expected behavior: The top 20% and bottom 20% should be replaced
        expected_winsorized_values = winsorize(self.data['feature1'], limits=(0.2, 0.2))
        
        # Check that Winsorized values match expected values
        pd.testing.assert_series_equal(winsorized_df['feature1'], pd.Series(expected_winsorized_values), check_names=False)

    def test_auto_transform(self):
        """Test the auto_transform method."""
        transformed_df = DataTransformer.auto_transform(self.data, target='target', log_threshold=0.5, outlier_threshold=1.5, winsorize_limits=(0.05, 0.05))
        
        # Verify log transformation
        log_transformed_df = DataTransformer.apply_log_transformation(self.data, 'feature1')
        self.assertTrue(np.all(transformed_df['feature1'] >= 0))
        
        # Verify outliers are imputed after log transformation
        log_transformed_imputed_df = DataTransformer.impute_outliers(log_transformed_df, 'feature1', method='median')
        self.assertTrue(np.all(transformed_df['feature1'] >= 0))

        # Verify Winsorizing is applied after log transformation and outlier imputation
        winsorized_log_transformed_imputed_df = DataTransformer.winsorize_data(log_transformed_imputed_df, 'feature1', limits=(0.05, 0.05))
        self.assertTrue(np.allclose(transformed_df['feature1'], winsorized_log_transformed_imputed_df['feature1']))

if __name__ == '__main__':
    unittest.main()
