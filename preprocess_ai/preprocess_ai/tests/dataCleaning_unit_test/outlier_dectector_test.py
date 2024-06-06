import unittest
import pandas as pd
from preprocess_ai.DataPreparation.DataCleaner.outlierHandler import OutlierDetector  # Correct import statement if your class is in 'outlierDetector.py'

class TestOutlierDetector(unittest.TestCase):
    def setUp(self):
        """Set up test variables for all tests."""
        self.data_simple = pd.DataFrame({
            'Values': [10, 12, 14, 15, 17, 19, 20, 30, 90, 100]
        })
        self.detector_simple = OutlierDetector(self.data_simple, 'Values')

        self.data_no_outliers = pd.DataFrame({
            'Values': [10, 11, 12, 13, 14, 15, 16, 17, 18, 19]
        })
        self.detector_no_outliers = OutlierDetector(self.data_no_outliers, 'Values')

        self.data_with_outlier = pd.DataFrame({
            'Values': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 100]
        })
        self.detector_with_outlier = OutlierDetector(self.data_with_outlier, 'Values')

    def test_calculate_iqr(self):
        """Test the IQR calculation."""
        Q1, Q3, IQR = self.detector_simple.calculate_iqr()
        self.assertEqual(Q1, 14.25)  # Corrected Q1 value
        self.assertEqual(Q3, 27.5)   # Q3 value remains the same
        self.assertEqual(IQR, 13.25)  # Corrected IQR value

    def test_find_outliers(self):
        """Test the detection of outliers."""
        outliers = self.detector_simple.find_outliers()
        self.assertEqual(len(outliers), 2)  # Expecting two outliers
        self.assertIn(90, outliers['Values'].values)
        self.assertIn(100, outliers['Values'].values)

    def test_no_outliers(self):
        """Test when there are no outliers."""
        outliers = self.detector_no_outliers.find_outliers()
        self.assertEqual(len(outliers), 0)  # Expecting no outliers

    def test_remove_outliers(self):
        """Test the removal of outliers."""
        cleaned_data = self.detector_with_outlier.remove_outliers()
        expected_values = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]  # Assuming 100 is the only outlier
        self.assertListEqual(list(cleaned_data['Values']), expected_values)

if __name__ == '__main__':
    unittest.main()
