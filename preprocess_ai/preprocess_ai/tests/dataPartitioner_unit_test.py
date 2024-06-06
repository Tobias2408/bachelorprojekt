import unittest
import pandas as pd
from pandas.testing import assert_frame_equal, assert_series_equal
from preprocess_ai.DataPreparation.DataPartitioner import DataPartitioner  # Update this import based on your file structure

class TestDataPartitioner(unittest.TestCase):

    def setUp(self):
        self.data = {
            'feature1': [10, 20, 30, 40, 50, 60, 70, 80, 90, 100],
            'feature2': [5, 10, 15, 20, 25, 30, 35, 40, 45, 50],
            'target': [0, 1, 0, 1, 0, 1, 0, 1, 0, 1]
        }
        self.df = pd.DataFrame(self.data)

    def test_split_data_stratified(self):
        partitioner = DataPartitioner()
        X_train, X_val, X_test, y_train, y_val, y_test = partitioner.split_data(
            self.df, target='target', test_size=0.2, val_size=0.2, stratify=True
        )
        
        # Check the shapes of the splits
        self.assertEqual(X_train.shape[0], 6)
        self.assertEqual(X_val.shape[0], 2)
        self.assertEqual(X_test.shape[0], 2)
        self.assertEqual(y_train.shape[0], 6)
        self.assertEqual(y_val.shape[0], 2)
        self.assertEqual(y_test.shape[0], 2)
        
        # Check the stratification
        self.assertAlmostEqual(y_train.mean(), 0.5, delta=0.1)
        self.assertAlmostEqual(y_val.mean(), 0.5, delta=0.1)
        self.assertAlmostEqual(y_test.mean(), 0.5, delta=0.1)

    def test_split_data_non_stratified(self):
        partitioner = DataPartitioner()
        X_train, X_val, X_test, y_train, y_val, y_test = partitioner.split_data(
            self.df, target='target', test_size=0.2, val_size=0.2, stratify=False
        )
        
        # Check the shapes of the splits
        self.assertEqual(X_train.shape[0], 6)
        self.assertEqual(X_val.shape[0], 2)
        self.assertEqual(X_test.shape[0], 2)
        self.assertEqual(y_train.shape[0], 6)
        self.assertEqual(y_val.shape[0], 2)
        self.assertEqual(y_test.shape[0], 2)
        
        # Stratification is not guaranteed, so just check the overall length
        self.assertEqual(len(y_train) + len(y_val) + len(y_test), 10)

    def test_split_data_val_size_too_small(self):
        partitioner = DataPartitioner()
        with self.assertRaises(ValueError):
            partitioner.split_data(
                self.df, target='target', test_size=0.2, val_size=0.05, stratify=True
            )

if __name__ == '__main__':
    unittest.main()
