import requests
import json
import unittest
from dotenv import load_dotenv
import os
import numpy as np
import pandas as pd
from sklearn.datasets import fetch_openml
from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker
from SimpleAI_Image import DatabaseHandler, DataProcessor
from tensorflow.keras.applications.vgg16 import preprocess_input as vgg_preprocess_input

# Load environment variables from .env file
load_dotenv()
API_KEY = os.getenv("OPENAI_API_KEY")  # Get the API key from .env

# Constants
API_URL = "https://api.openai.com/v1/embeddings"
HEADERS = {
    "Content-Type": "application/json",
    "Authorization": f"Bearer {API_KEY}"
}

# Define the database URL
db_url = 'postgresql+psycopg2://tobiaspoulsen:Bubber240811@localhost:5432/ThisISATEST'
engine = create_engine(db_url)
Session = sessionmaker(bind=engine)
session = Session()

# Instantiate DataProcessor with VGG16 model
data_processor = DataProcessor(DatabaseHandler(db_url, 'vector_data', 512), model_name='VGG16', preprocess_func=vgg_preprocess_input, image_size=(32, 32))

class TestEmbeddingAPI(unittest.TestCase):

    def setUp(self):
        # Create a temporary table for testing
        session.execute(text("""
            CREATE TEMP TABLE temp_vector_data AS
            SELECT * FROM vector_data WHERE 1=0
        """))
        session.commit()

        # Update the DataProcessor to use the temporary table
        self.db_handler = DatabaseHandler(db_url, 'temp_vector_data', 512)
        self.data_processor = DataProcessor(self.db_handler, model_name='VGG16', preprocess_func=vgg_preprocess_input, image_size=(32, 32))

    def tearDown(self):
        # Drop the temporary table after each test
        session.execute(text("DROP TABLE IF EXISTS temp_vector_data"))
        session.commit()

    def test_generate_embedding_for_simple_text_string(self):
        data = {
            "input": "This is a test.",
            "model": "text-embedding-3-small"
        }
        response = requests.post(API_URL, headers=HEADERS, data=json.dumps(data))
        self.assertEqual(response.status_code, 200, "API request failed")
        response_json = response.json()
        self.assertIn("data", response_json, "No 'data' field in response")
        self.assertIsInstance(response_json["data"], list, "'data' field is not a list")
        self.assertGreater(len(response_json["data"]), 0, "'data' list is empty")
        embedding = response_json["data"][0].get("embedding")
        self.assertIsNotNone(embedding, "No 'embedding' field in data")
        self.assertIsInstance(embedding, list, "'embedding' is not a list")
        self.assertTrue(all(isinstance(x, float) for x in embedding), "Not all elements in 'embedding' are floats")
        print("Test TC01 passed successfully.")

    def test_generate_embedding_for_large_text_string(self):
        large_text = "This is a large text. " * 1000
        data = {
            "input": large_text,
            "model": "text-embedding-3-small"
        }
        response = requests.post(API_URL, headers=HEADERS, data=json.dumps(data))
        self.assertEqual(response.status_code, 200, "API request failed")
        response_json = response.json()
        self.assertIn("data", response_json, "No 'data' field in response")
        self.assertIsInstance(response_json["data"], list, "'data' field is not a list")
        self.assertGreater(len(response_json["data"]), 0, "'data' list is empty")
        embedding = response_json["data"][0].get("embedding")
        self.assertIsNotNone(embedding, "No 'embedding' field in data")
        self.assertIsInstance(embedding, list, "'embedding' is not a list")
        self.assertTrue(all(isinstance(x, float) for x in embedding), "Not all elements in 'embedding' are floats")
        print("Test TC02 passed successfully.")

    def test_invalid_api_key(self):
        headers = {
            "Content-Type": "application/json",
            "Authorization": "Bearer invalid_api_key"
        }
        data = {
            "input": "This is a test.",
            "model": "text-embedding-3-small"
        }
        response = requests.post(API_URL, headers=headers, data=json.dumps(data))
        self.assertNotEqual(response.status_code, 200, "API request did not fail as expected")
        response_json = response.json()
        self.assertIn("error", response_json, "No 'error' field in response")
        print("Test TC03 passed successfully.")

    def test_handling_special_characters(self):
        data = {
            "input": "!@#$%^&*()_+-=[]{}|;':,.<>/?`~",
            "model": "text-embedding-3-small"
        }
        response = requests.post(API_URL, headers=HEADERS, data=json.dumps(data))
        self.assertEqual(response.status_code, 200, "API request failed")
        response_json = response.json()
        self.assertIn("data", response_json, "No 'data' field in response")
        self.assertIsInstance(response_json["data"], list, "'data' field is not a list")
        self.assertGreater(len(response_json["data"]), 0, "'data' list is empty")
        embedding = response_json["data"][0].get("embedding")
        self.assertIsNotNone(embedding, "No 'embedding' field in data")
        self.assertIsInstance(embedding, list, "'embedding' is not a list")
        self.assertTrue(all(isinstance(x, float) for x in embedding), "Not all elements in 'embedding' are floats")
        print("Test TC04 passed successfully.")

    def test_embedding_with_different_models(self):
        data_small = {
            "input": "This is a test.",
            "model": "text-embedding-3-small"
        }
        data_large = {
            "input": "This is a test.",
            "model": "text-embedding-3-large"
        }
        response_small = requests.post(API_URL, headers=HEADERS, data=json.dumps(data_small))
        response_large = requests.post(API_URL, headers=HEADERS, data=json.dumps(data_large))
        self.assertEqual(response_small.status_code, 200, "API request failed for small model")
        self.assertEqual(response_large.status_code, 200, "API request failed for large model")
        embedding_small = response_small.json()["data"][0].get("embedding")
        embedding_large = response_large.json()["data"][0].get("embedding")
        self.assertIsInstance(embedding_small, list, "'embedding' for small model is not a list")
        self.assertIsInstance(embedding_large, list, "'embedding' for large model is not a list")
        self.assertTrue(all(isinstance(x, float) for x in embedding_small), "Not all elements in 'embedding' for small model are floats")
        self.assertTrue(all(isinstance(x, float) for x in embedding_large), "Not all elements in 'embedding' for large model are floats")
        print("Test TC05 passed successfully.")

    def test_performance_under_load(self):
        import time
        start_time = time.time()
        for _ in range(10):
            data = {
                "input": "This is a test.",
                "model": "text-embedding-3-small"
            }
            response = requests.post(API_URL, headers=HEADERS, data=json.dumps(data))
            self.assertEqual(response.status_code, 200, "API request failed under load")
        end_time = time.time()
        total_time = end_time - start_time
        self.assertLess(total_time, 10, "API performance is not within acceptable limits")
        print("Test TC06 passed successfully.")

    def test_integration_with_vector_database(self):
        # Load the example dataset (MNIST)
        mnist = fetch_openml('mnist_784', version=1)
        X = mnist.data[:500]  # Limit to 500 instances for testing
        y = mnist.target[:500].astype(int)  # Ensure targets are integers

        # Process data to extract features
        X_embedded, _ = self.data_processor.process_data(X, y)

        # Store vectors in the temporary table without dimensionality reduction
        self.db_handler.store_vectors(X_embedded, y)

        # Fetch and preprocess data for visualization
        query = text("SELECT * FROM temp_vector_data")
        X_embedded, y = self.data_processor.fetch_and_preprocess_data(query)

        # Verify that the data was fetched and preprocessed correctly
        self.assertIsNotNone(X_embedded, "No embedded data retrieved from the database")
        self.assertIsNotNone(y, "No labels retrieved from the database")
        self.assertTrue(len(X_embedded) > 0, "Embedded data retrieved from the database is empty")
        self.assertTrue(len(y) > 0, "Labels retrieved from the database are empty")
        
        # Visualize data (optional for the test, mainly for manual verification)
        # self.data_processor.visualize_data(X_embedded, y)
        print("Test TC07 passed successfully.")

    def test_token_count_calculation(self):
        import tiktoken
        text = "This is a test."
        encoding = tiktoken.get_encoding("cl100k_base")
        num_tokens = len(encoding.encode(text))
        data = {
            "input": text,
            "model": "text-embedding-3-small"
        }
        response = requests.post(API_URL, headers=HEADERS, data=json.dumps(data))
        self.assertEqual(response.status_code, 200, "API request failed")
        response_json = response.json()
        self.assertEqual(response_json["usage"]["total_tokens"], num_tokens, "Token count does not match")
        print("Test TC08 passed successfully.")

    def test_embedding_reduction(self):
        data = {
            "input": "This is a test.",
            "model": "text-embedding-3-small",
            "dimensions": 512
        }
        response = requests.post(API_URL, headers=HEADERS, data=json.dumps(data))
        self.assertEqual(response.status_code, 200, "API request failed")
        response_json = response.json()
        embedding = response_json["data"][0].get("embedding")
        self.assertIsNotNone(embedding, "No 'embedding' field in data")
        self.assertIsInstance(embedding, list, "'embedding' is not a list")
        self.assertEqual(len(embedding), 512, "Embedding dimensions do not match requested dimensions")
        self.assertTrue(all(isinstance(x, float) for x in embedding), "Not all elements in 'embedding' are floats")
        print("Test TC09 passed successfully.")

    def test_handling_missing_input(self):
        data = {
            "model": "text-embedding-3-small"
        }
        response = requests.post(API_URL, headers=HEADERS, data=json.dumps(data))
        self.assertNotEqual(response.status_code, 200, "API request did not fail as expected")
        response_json = response.json()
        self.assertIn("error", response_json, "No 'error' field in response")
        print("Test TC10 passed successfully.")

if __name__ == "__main__":
    unittest.main()
