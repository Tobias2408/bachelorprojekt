import unittest
import pandas as pd
from pandas.testing import assert_frame_equal
from DataPreparation.DataTransformation.categoricalEncoder import CategoricalEncoder

class TestCategoricalEncoder(unittest.TestCase):

    def setUp(self):
        self.data = {'color': ['red', 'blue', 'green', 'blue'], 'size': ['S', 'M', 'L', 'S']}
        self.df = pd.DataFrame(self.data)

    def test_fit_transform_label_encoding(self):
        expected_data = {'color': [2, 0, 1, 0], 'size': ['S', 'M', 'L', 'S']}
        expected_df = pd.DataFrame(expected_data)
        
        df, encoders = CategoricalEncoder.fit_transform_label_encoding(self.df.copy(), ['color'])
        
        assert_frame_equal(df, expected_df)
        self.assertIn('color', encoders)

    def test_transform_label_encoding(self):
        _, encoders = CategoricalEncoder.fit_transform_label_encoding(self.df.copy(), ['color'])
        new_data = {'color': ['green', 'blue', 'red'], 'size': ['L', 'S', 'M']}
        new_df = pd.DataFrame(new_data)
        expected_data = {'color': [1, 0, 2], 'size': ['L', 'S', 'M']}
        expected_df = pd.DataFrame(expected_data)
        
        transformed_df = CategoricalEncoder.transform_label_encoding(new_df.copy(), ['color'], encoders)
        
        assert_frame_equal(transformed_df, expected_df)

    def test_fit_transform_onehot_encoding(self):
        expected_data = {'color': ['red', 'blue', 'green', 'blue'],
                         'size_S': [1, 0, 0, 1], 'size_M': [0, 1, 0, 0]}
        expected_df = pd.DataFrame(expected_data)
        
        df, encoders, onehot_encoded_cols = CategoricalEncoder.fit_transform_onehot_encoding(self.df.copy(), ['size'])
        
        # Ensure the columns are in the same order before comparison
        df = df[expected_df.columns]
        assert_frame_equal(df, expected_df)
        self.assertIn('size', encoders)
        self.assertListEqual(sorted(onehot_encoded_cols), sorted(['size_S', 'size_M']))

    def test_transform_onehot_encoding(self):
        _, encoders, _ = CategoricalEncoder.fit_transform_onehot_encoding(self.df.copy(), ['size'])
        new_data = {'color': ['green', 'blue', 'red'], 'size': ['L', 'S', 'M']}
        new_df = pd.DataFrame(new_data)
        expected_data = {'color': ['green', 'blue', 'red'],
                         'size_S': [0, 1, 0], 'size_M': [0, 0, 1]}
        expected_df = pd.DataFrame(expected_data)
        
        transformed_df = CategoricalEncoder.transform_onehot_encoding(new_df.copy(), ['size'], encoders)
        
        # Ensure the columns are in the same order before comparison
        transformed_df = transformed_df[expected_df.columns]
        assert_frame_equal(transformed_df, expected_df)

if __name__ == '__main__':
    unittest.main()
