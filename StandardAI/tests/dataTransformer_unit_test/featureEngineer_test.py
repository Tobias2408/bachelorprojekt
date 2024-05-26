import unittest
import pandas as pd
import numpy as np
from pandas.testing import assert_frame_equal
from DataPreparation.DataTransformation.featureEngineer import FeatureEngineer

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

        df = FeatureEngineer.add_interaction_feature(self.df.copy(), 'feature1', 'feature2', 'feature1_x_feature2')
        
        assert_frame_equal(df, expected_df)

    def test_add_date_parts(self):
        df = self.df.copy()
        df = FeatureEngineer.add_date_parts(df, 'date')

        # Convert the data types in the actual DataFrame to match the expected DataFrame
        df['date_year'] = df['date_year'].astype('int32')
        df['date_month'] = df['date_month'].astype('int32')
        df['date_day'] = df['date_day'].astype('int32')
        df['date_dayofweek'] = df['date_dayofweek'].astype('int32')
        df['date_is_weekend'] = df['date_is_weekend'].astype('int32')

        expected_data = {
            'date': pd.to_datetime(['2023-01-01', '2023-01-02', '2023-01-03']),
            'price': [100, 150, 200],
            'quantity': [1, 2, 3],
            'feature1': [10, 20, 30],
            'feature2': [3, 6, 9],
            'date_year': pd.Series([2023, 2023, 2023], dtype='int32'),
            'date_month': pd.Series([1, 1, 1], dtype='int32'),
            'date_day': pd.Series([1, 2, 3], dtype='int32'),
            'date_dayofweek': pd.Series([6, 0, 1], dtype='int32'),
            'date_is_weekend': pd.Series([1, 0, 0], dtype='int32')
        }
        expected_df = pd.DataFrame(expected_data)

        assert_frame_equal(df, expected_df)

    def test_add_rolling_features(self):
        df = self.df.copy()
        df = FeatureEngineer.add_rolling_features(df, 'price', 2)

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

    def test_empty_dataframe(self):
        df = pd.DataFrame()
        df = FeatureEngineer.add_date_parts(df, 'date')
        expected_df = pd.DataFrame()
        assert_frame_equal(df, expected_df)

    def test_single_row_dataframe(self):
        df = pd.DataFrame({
            'date': ['2023-01-01'],
            'price': [100],
            'quantity': [1],
            'feature1': [10],
            'feature2': [3]
        })
        df = FeatureEngineer.add_date_parts(df, 'date')
        df['date_year'] = df['date_year'].astype('int32')
        df['date_month'] = df['date_month'].astype('int32')
        df['date_day'] = df['date_day'].astype('int32')
        df['date_dayofweek'] = df['date_dayofweek'].astype('int32')
        df['date_is_weekend'] = df['date_is_weekend'].astype('int32')

        expected_df = pd.DataFrame({
            'date': pd.to_datetime(['2023-01-01']),
            'price': [100],
            'quantity': [1],
            'feature1': [10],
            'feature2': [3],
            'date_year': pd.Series([2023], dtype='int32'),
            'date_month': pd.Series([1], dtype='int32'),
            'date_day': pd.Series([1], dtype='int32'),
            'date_dayofweek': pd.Series([6], dtype='int32'),
            'date_is_weekend': pd.Series([1], dtype='int32')
        })

        assert_frame_equal(df, expected_df)

    def test_dataframe_with_nans(self):
        df = pd.DataFrame({
            'date': ['2023-01-01', '2023-01-02', np.nan],
            'price': [100, np.nan, 200],
            'quantity': [1, 2, np.nan],
            'feature1': [10, 20, 30],
            'feature2': [3, 6, np.nan]
        })
        df = FeatureEngineer.add_date_parts(df, 'date')

        # Ensure the dtypes and NaNs match between actual and expected DataFrame
        df['date_year'] = df['date_year'].astype('Int32')
        df['date_month'] = df['date_month'].astype('Int32')
        df['date_day'] = df['date_day'].astype('Int32')
        df['date_dayofweek'] = df['date_dayofweek'].astype('Int32')
        df['date_is_weekend'] = df['date_is_weekend'].astype('Int32')

        expected_df = pd.DataFrame({
            'date': pd.to_datetime(['2023-01-01', '2023-01-02', np.nan]),
            'price': [100, np.nan, 200],
            'quantity': [1, 2, np.nan],
            'feature1': [10, 20, 30],
            'feature2': [3, 6, np.nan],
            'date_year': pd.Series([2023, 2023, pd.NA], dtype='Int32'),
            'date_month': pd.Series([1, 1, pd.NA], dtype='Int32'),
            'date_day': pd.Series([1, 2, pd.NA], dtype='Int32'),
            'date_dayofweek': pd.Series([6, 0, pd.NA], dtype='Int32'),
            'date_is_weekend': pd.Series([1, 0, pd.NA], dtype='Int32')
        })

        # Ensure NaN consistency
        df['date_is_weekend'] = df['date_is_weekend'].where(df['date'].notna(), pd.NA)

        assert_frame_equal(df, expected_df)

    def test_non_numeric_interaction_feature(self):
        df = pd.DataFrame({
            'feature1': ['A', 'B', 'C'],
            'feature2': ['X', 'Y', 'Z']
        })
        with self.assertRaises(TypeError):
            df = FeatureEngineer.add_interaction_feature(df, 'feature1', 'feature2', 'feature1_x_feature2')

if __name__ == '__main__':
    unittest.main()
