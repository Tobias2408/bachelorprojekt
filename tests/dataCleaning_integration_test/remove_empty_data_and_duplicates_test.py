import unittest
import pandas as pd
from DataPreparation.DataCleaner.dataCleaner import dataCleaner  # Ensure this is the correct import path
from DataPreparation.DataCleaner.duplicationHandler import DuplicationHandler

class RemoveEmptyDataAndDuplicatesTest(unittest.TestCase):
    def setUp(self):
        data = {
            'Feature1': [1, 1, 2, None, 4, 4],
            'Feature2': [4, 4, 2, None, 2, 1],
            'Feature3': ['A', 'A', 'B', 'C', 'B', 'C'],
            'Label': [10, 10, 11, None, 12, 13]
        }
        self.df = pd.DataFrame(data)

    def test_smart_clean_and_remove_duplicates(self):
        cleaned_df = dataCleaner().smart_clean_data(self.df, 'Label', 0.3, 0.5)
        print(cleaned_df)
        final_df = DuplicationHandler.remove_duplicates(cleaned_df)
        print(final_df)
        expected_data = {
            'Feature1': [1.0, 2.0, 4.0, 4.0],
            'Feature2': [4.0, 2.0, 2.0, 1.0],
            'Feature3': ['A', 'B', 'B', 'C'],
            'Label': [10.0, 11.0, 12.0, 13.0]
        }
        expected_df = pd.DataFrame(expected_data)

        pd.testing.assert_frame_equal(
            final_df.reset_index(drop=True),
            expected_df.reset_index(drop=True)
        )

if __name__ == '__main__':
    unittest.main()
