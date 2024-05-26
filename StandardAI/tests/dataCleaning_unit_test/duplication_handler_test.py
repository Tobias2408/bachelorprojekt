import unittest
import pandas as pd
import os
import sys

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from DataPreparation.DataCleaner.duplicationHandler import DuplicationHandler  # type: ignore # Assuming the class is saved in a file named duplication_handler.py

class TestDuplicationHandler(unittest.TestCase):
    def setUp(self):
        """Create a sample DataFrame to use in tests."""
        self.data = {
            'Name': ['Alice', 'Bob', 'Alice', 'Charlie', 'Bob', 'Charlie', 'Dave'],
            'Score': [85, 90, 85, 88, 90, 88, 100],
            'Age': [25, 30, 25, 35, 30, 35, 40]
        }
        self.df = pd.DataFrame(self.data)
    
    def test_identify_duplicates(self):
        expected_duplicates = pd.DataFrame({
            'Name': ['Alice', 'Alice', 'Bob', 'Bob', 'Charlie', 'Charlie'],
            'Score': [85, 85, 90, 90, 88, 88],
            'Age': [25, 25, 30, 30, 35, 35]
        }).sort_values(by=['Name', 'Score', 'Age']).reset_index(drop=True)  # Ensure sorting

        result_df = DuplicationHandler.identify_duplicates(self.df, subset=['Name', 'Score'])
        result_df = result_df.sort_values(by=['Name', 'Score', 'Age']).reset_index(drop=True)  # Ensure sorting

        pd.testing.assert_frame_equal(result_df, expected_duplicates)

    def test_remove_duplicates(self):
        result_df = DuplicationHandler.remove_duplicates(self.df.copy(), subset=['Name', 'Score'])
        expected_data = {
            'Name': ['Alice', 'Bob', 'Charlie', 'Dave'],  # Remove second 'Charlie' from expected data
            'Score': [85, 90, 88, 100],
            'Age': [25, 30, 35, 40]  # Corresponding ages
        }
        expected_df = pd.DataFrame(expected_data)
        
        pd.testing.assert_frame_equal(result_df.reset_index(drop=True), expected_df.reset_index(drop=True))


    def test_log_duplicates(self):
        """Test that duplicates are correctly logged to a CSV file."""
        DuplicationHandler.log_duplicates(self.df, subset=['Name', 'Score'], filepath='test_duplicates_log.csv')
        logged_df = pd.read_csv('test_duplicates_log.csv')
        expected_logged = DuplicationHandler.identify_duplicates(self.df, subset=['Name', 'Score'])
        pd.testing.assert_frame_equal(logged_df.sort_values(by=['Name', 'Score', 'Age']).reset_index(drop=True), 
                                      expected_logged.sort_values(by=['Name', 'Score', 'Age']).reset_index(drop=True))

    def test_aggregate_duplicates(self):
        aggregated_df = DuplicationHandler.aggregate_duplicates(self.df, key_columns=['Name'], agg_column='Score', agg_func='mean')
        expected_aggregated = pd.DataFrame({
            'Name': ['Alice', 'Bob', 'Charlie', 'Dave'],
            'Score': [85.0, 90.0, 88.0, 100.0]  # Ensure these are floats if using 'mean'
        })
        pd.testing.assert_frame_equal(aggregated_df.reset_index(drop=True), expected_aggregated.reset_index(drop=True))

if __name__ == '__main__':
    unittest.main()
