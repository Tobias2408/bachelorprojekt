import unittest
import pandas as pd
from pandas.testing import assert_frame_equal
from sklearn.feature_selection import f_classif, f_regression
from preprocess_ai.DataPreparation.DataReduction.featureSelector import FeatureSelector  # Update this import based on your file structure

class TestFeatureSelector(unittest.TestCase):

    def setUp(self):
        self.data = {
            'feature1': [10, 20, 30, 40, 50],
            'feature2': [5, 10, 15, 20, 25],
            'feature3': [1, 2, 3, 4, 5],
            'target': [0, 1, 0, 1, 0]
        }
        self.df = pd.DataFrame(self.data)
        self.target = self.df.pop('target')

    def test_select_k_best(self):
        df_selected = FeatureSelector.select_k_best(self.df, self.target, k=2, score_func=f_classif)
        self.assertEqual(df_selected.shape[1], 2)

    def test_select_by_model(self):
        df_selected = FeatureSelector.select_by_model(self.df, self.target, n_features_to_select=2)
        self.assertEqual(df_selected.shape[1], 2)

    def test_select_by_rfe(self):
        df_selected = FeatureSelector.select_by_rfe(self.df, self.target, n_features_to_select=2)
        self.assertEqual(df_selected.shape[1], 2)

if __name__ == '__main__':
    unittest.main()
