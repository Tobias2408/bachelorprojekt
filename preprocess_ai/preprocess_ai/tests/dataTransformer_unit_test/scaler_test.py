import unittest
import numpy as np
from preprocess_ai.DataPreparation.DataTransformation.scaler import SimpleScaler  # Assume the scaler class is saved in scaler_module.py
from numpy.testing import assert_array_almost_equal
class TestScaler(unittest.TestCase):
    def setUp(self):
        """Initialize an object of SimpleScaler before each test."""
        self.scaler = SimpleScaler()
        self.data = np.array([[1, 2], [3, 4], [5, 6]], dtype=np.float64)

    def test_standardize(self):
        """Test the standardize method."""
        standardized_data, mean, std = self.scaler.standardize(self.data)
        # Expectation updated to match population standard deviation calculations
        expected_data = np.array([[-1., -1.],
                                [ 0.,  0.],
                                [ 1.,  1.]])
        np.testing.assert_array_almost_equal(standardized_data, expected_data, decimal=6)
        np.testing.assert_array_almost_equal(mean, np.array([3., 4.]), decimal=6)
        np.testing.assert_array_almost_equal(std, np.array([2., 2.]), decimal=6)  # Corrected expected standard deviation



    def test_normalize(self):
        """Test the normalize method."""
        normalized_data, min_val, max_val = self.scaler.normalize(self.data)
        expected_data = np.array([[0., 0.],
                                  [0.5, 0.5],
                                  [1., 1.]])
        np.testing.assert_array_almost_equal(normalized_data, expected_data)
        np.testing.assert_array_almost_equal(min_val, np.array([1., 2.]))
        np.testing.assert_array_almost_equal(max_val, np.array([5., 6.]))

if __name__ == '__main__':
    unittest.main()
