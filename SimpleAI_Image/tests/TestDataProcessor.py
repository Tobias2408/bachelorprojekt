import unittest
from unittest.mock import MagicMock
import numpy as np
import pandas as pd
from SimpleAI_Image import DataProcessor  # Adjust the import to your module's structure
# Test uses file directly not pip
class TestDataProcessor(unittest.TestCase):
    def setUp(self):
        # Create a mock db_handler
        self.mock_db_handler = MagicMock()

        # Mock the methods you expect to call on db_handler
        self.mock_db_handler.fetch_data.return_value = pd.DataFrame({
            'features': ['[0.1, 0.2, 0.3]', '[0.4, 0.5, 0.6]'],
            'label': [1, 2]
        })
        self.mock_db_handler.store_vectors = MagicMock()

        # Define a dummy preprocessing function
        def dummy_preprocess_func(images):
            return images / 255.0
        
        # Create an instance of DataProcessor with the mock db_handler
        self.processor = DataProcessor(
            db_handler=self.mock_db_handler,
            preprocess_func=dummy_preprocess_func
        )

    def test_fetch_and_preprocess_data(self):
        query = "SELECT * FROM some_table"
        X_embedded, y = self.processor.fetch_and_preprocess_data(query)
        
        # Check the fetch_data method was called once with the query
        self.mock_db_handler.fetch_data.assert_called_once_with(query)

        # Check the output shapes
        self.assertEqual(X_embedded.shape, (2, 3))
        self.assertEqual(list(y), [1, 2])

    def test_process_data(self):
        # Create dummy data
        X = pd.DataFrame({
            'image': [np.random.rand(28 * 28) for _ in range(2)]
        })
        y = pd.Series([1, 2])

        # Mock the preprocess_func to fit our dummy data
        def dummy_preprocess_func(images):
            return images / 255.0

        # Update the processor with this new preprocess function
        self.processor.preprocess_func = dummy_preprocess_func

        # Process the data
        X_embedded, y = self.processor.process_data(X, y)

        # Check if the store_vectors method was called
        self.mock_db_handler.store_vectors.assert_called_once()

        # Check the output shapes
        self.assertEqual(X_embedded.shape[0], X.shape[0])
        self.assertEqual(list(y), [1, 2])

    def test_empty_data(self):
        # Edge case: Empty DataFrame and Series
        X = pd.DataFrame({'image': []})
        y = pd.Series([], dtype=int)

        with self.assertRaises(ValueError):
            self.processor.process_data(X, y)

    def test_single_image(self):
        # Edge case: Single image input
        X = pd.DataFrame({
            'image': [np.random.rand(28 * 28)]
        })
        y = pd.Series([1])

        # Process the data
        X_embedded, y = self.processor.process_data(X, y)

        # Check the output shapes
        self.assertEqual(X_embedded.shape[0], 1)
        self.assertEqual(list(y), [1])

    def test_large_dataset(self):
        # Edge case: Large dataset
        num_samples = 1000
        X = pd.DataFrame({
            'image': [np.random.rand(28 * 28) for _ in range(num_samples)]
        })
        y = pd.Series(np.random.randint(0, 10, size=num_samples))

        # Process the data
        X_embedded, y = self.processor.process_data(X, y)

        # Check the output shapes
        self.assertEqual(X_embedded.shape[0], num_samples)
        self.assertEqual(len(y), num_samples)

if __name__ == '__main__':
    unittest.main()
