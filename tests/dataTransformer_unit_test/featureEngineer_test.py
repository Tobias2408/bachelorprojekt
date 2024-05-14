import unittest
import pandas as pd
import numpy as np
from pandas.testing import assert_frame_equal
from DataPreparation.DataTransformation.featureEngineer import FeatureEngineer, interaction_feature, date_parts, rolling_features, total_sales

class TestFeatureEngineer(unittest.TestCase):

    def setUp(self):
        self.data = {
            'date': ['2023-01-01', '2023-01-02', '2023-01-03'],
            'price': [100, 150, 200],
            'quantity': [1, 2, 3],
            'feature1': [10, 20, 30],
            'feature2': [3, 6, 9]
        }
        self.df = pd.DataFrame(self.data)

    def test_add_interaction_feature(self):
        expected_data = {
            'date': ['2023-01-01', '2023-01-02', '2023-01-03'],
            'price': [100, 150, 200],
            'quantity': [1, 2, 3],
            'feature1': [10, 20, 30],
            'feature2': [3, 6, 9],
            'feature1_x_feature2': [30, 120, 270]
        }
        expected_df = pd.DataFrame(expected_data)

        df = FeatureEngineer.add_custom_feature(self.df.copy(), 'feature1_x_feature2', interaction_feature, 'feature1', 'feature2')
        
        assert_frame_equal(df, expected_df)

    def test_add_date_parts(self):
        df = self.df.copy()
        date_parts_df = date_parts(df, 'date')
        df = df.join(date_parts_df)

        # Convert integer columns in df to int64
        df['date_year'] = df['date_year'].astype('int64')
        df['date_month'] = df['date_month'].astype('int64')
        df['date_day'] = df['date_day'].astype('int64')
        df['date_dayofweek'] = df['date_dayofweek'].astype('int64')
        df['date_is_weekend'] = df['date_is_weekend'].astype('int64')

        expected_data = {
            'date': pd.to_datetime(['2023-01-01', '2023-01-02', '2023-01-03']),
            'price': [100, 150, 200],
            'quantity': [1, 2, 3],
            'feature1': [10, 20, 30],
            'feature2': [3, 6, 9],
            'date_year': pd.Series([2023, 2023, 2023], dtype='int64'),
            'date_month': pd.Series([1, 1, 1], dtype='int64'),
            'date_day': pd.Series([1, 2, 3], dtype='int64'),
            'date_dayofweek': pd.Series([6, 0, 1], dtype='int64'),
            'date_is_weekend': pd.Series([1, 0, 0], dtype='int64')
        }
        expected_df = pd.DataFrame(expected_data)

        assert_frame_equal(df, expected_df)

    def test_add_rolling_features(self):
        df = self.df.copy()
        rolling_features_df = rolling_features(df, 'price', window_size=2)
        df = df.join(rolling_features_df)

        expected_data = {
            'date': ['2023-01-01', '2023-01-02', '2023-01-03'],
            'price': [100, 150, 200],
            'quantity': [1, 2, 3],
            'feature1': [10, 20, 30],
            'feature2': [3, 6, 9],
            'price_rolling_mean_2': [np.nan, 125.0, 175.0],
            'price_rolling_std_2': [np.nan, 35.35533905932738, 35.35533905932738]
        }
        expected_df = pd.DataFrame(expected_data)

        assert_frame_equal(df, expected_df)

    def test_add_total_sales(self):
        expected_data = {
            'date': ['2023-01-01', '2023-01-02', '2023-01-03'],
            'price': [100, 150, 200],
            'quantity': [1, 2, 3],
            'feature1': [10, 20, 30],
            'feature2': [3, 6, 9],
            'total_sales': [100, 300, 600]
        }
        expected_df = pd.DataFrame(expected_data)

        df = FeatureEngineer.add_custom_feature(self.df.copy(), 'total_sales', total_sales, 'price', 'quantity')
        
        assert_frame_equal(df, expected_df)

if __name__ == '__main__':
    unittest.main()
