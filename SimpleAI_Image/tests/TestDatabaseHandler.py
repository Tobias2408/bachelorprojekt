import unittest
import numpy as np
import pandas as pd
import os
from sqlalchemy.exc import IntegrityError, ProgrammingError, StatementError
from sqlalchemy import create_engine, text
from SimpleAI_Image import DatabaseHandler  # Ensure the correct import path
from dotenv import load_dotenv
# Test uses file directly not pip

# Load environment variables from .env file
load_dotenv()

class TestDatabaseHandler(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.db_url = os.getenv('TEST_DB_URL')
        if cls.db_url is None:
            raise ValueError("TEST_DB_URL environment variable is not set.")
        cls.vector_size = 512

    def setUp(self):
        # Use a unique name for the temporary table to avoid conflicts
        self.table_name = 'test_table_' + os.urandom(4).hex()
        self.engine = create_engine(self.db_url)
        self.connection = self.engine.connect()
        self.trans = self.connection.begin()
        self.connection.execute(text('CREATE EXTENSION IF NOT EXISTS vector'))
        self.connection.execute(text(f'''
            CREATE TEMPORARY TABLE {self.table_name} (
                id SERIAL PRIMARY KEY,
                features VECTOR({self.vector_size}),
                label INTEGER
            );
        '''))
        self.trans.commit()  # Commit the creation of the temporary table
        self.db_handler = DatabaseHandler(self.db_url, self.table_name, self.vector_size)

    def tearDown(self):
        self.trans.rollback()
        self.connection.execute(text(f'DROP TABLE IF EXISTS {self.table_name}'))
        self.connection.close()

    def test_store_vectors_with_correct_input(self):
        vectors = [np.random.rand(self.vector_size), np.random.rand(self.vector_size)]
        labels = [1, 2]
        self.db_handler.store_vectors(vectors, labels)
        
        result = self.db_handler.fetch_data(f'SELECT * FROM {self.table_name}')
        self.assertEqual(len(result), 2)

    def test_store_vectors_with_empty_vectors(self):
        vectors = []
        labels = []
        self.db_handler.store_vectors(vectors, labels)
        
        result = self.db_handler.fetch_data(f'SELECT * FROM {self.table_name}')
        self.assertEqual(len(result), 0)

    def test_store_vectors_with_mismatched_lengths(self):
        vectors = [np.random.rand(self.vector_size)]
        labels = [1, 2]
        
        with self.assertRaises(ValueError):
            self.db_handler.store_vectors(vectors, labels)

    def test_store_vectors_with_large_vectors(self):
        large_vector = np.random.rand(self.vector_size + 1)
        vectors = [large_vector]
        labels = [1]
        
        with self.assertRaises(StatementError):  # Catching the correct exception
            self.db_handler.store_vectors(vectors, labels)

    def test_fetch_data_with_invalid_query(self):
        invalid_query = 'SELECT * FROM non_existent_table'
        
        with self.assertRaises(ProgrammingError):
            self.db_handler.fetch_data(invalid_query)

    def test_store_vectors_with_non_numeric_data(self):
        vectors = [np.array(['a', 'b', 'c'])]
        labels = [1]
        
        with self.assertRaises(StatementError):  # Catching the correct exception
            self.db_handler.store_vectors(vectors, labels)

    def test_store_vectors_with_null_label(self):
        vectors = [np.random.rand(self.vector_size)]
        labels = [None]
        
        with self.assertRaises(ValueError):  # Catching the correct exception
            self.db_handler.store_vectors(vectors, labels)

if __name__ == '__main__':
    unittest.main()
