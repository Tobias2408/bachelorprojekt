import unittest
import pandas as pd
from sklearn.decomposition import PCA
from pandas.testing import assert_frame_equal
from DataPreparation.DataReduction.dimensionalityReducer import DimensionalityReducer  # Update this import based on your file structure

class TestDimensionalityReducer(unittest.TestCase):

    def setUp(self):
        self.data = {
            'feature1': [10, 20, 30, 40, 50],
            'feature2': [5, 10, 15, 20, 25],
            'feature3': [1, 2, 3, 4, 5]
        }
        self.df = pd.DataFrame(self.data)

    def test_fit_pca(self):
        pca, scaler = DimensionalityReducer.fit_pca(self.df, n_components=2)
        self.assertIsInstance(pca, PCA)
        self.assertIsNotNone(scaler)

    def test_transform_pca(self):
        pca, scaler = DimensionalityReducer.fit_pca(self.df, n_components=2)
        df_pca = DimensionalityReducer.transform_pca(self.df, pca, scaler)
        self.assertEqual(df_pca.shape[1], 2)

    def test_fit_transform_pca(self):
        df_pca, pca, scaler = DimensionalityReducer.fit_transform_pca(self.df, n_components=2)
        self.assertEqual(df_pca.shape[1], 2)
        self.assertIsInstance(pca, PCA)
        self.assertIsNotNone(scaler)

    def test_pca_no_scaling(self):
        df_pca, pca, scaler = DimensionalityReducer.fit_transform_pca(self.df, n_components=2, scale_data=False)
        self.assertEqual(df_pca.shape[1], 2)
        self.assertIsInstance(pca, PCA)
        self.assertIsNone(scaler)

if __name__ == '__main__':
    unittest.main()
