import pandas as pd
from sklearn.datasets import fetch_openml
from sqlalchemy import text
from SimpleAI_Image import DatabaseHandler
from SimpleAI_Image import DataProcessor

def main():
    # Define the database URL and table name
    db_url = 'postgresql+psycopg2://tobiaspoulsen:Bubber240811@localhost:5432/ThisISATEST'
    table_name = 'vector_data'

    # Create database handler
    db_handler = DatabaseHandler(db_url, table_name, 512)

    # Instantiate DataProcessor with VGG16 model
    data_processor = DataProcessor(db_handler, model_name='VGG16', preprocess_func=vgg_preprocess_input, image_size=(32, 32))

    # Load the example dataset (MNIST)
    mnist = fetch_openml('mnist_784', version=1)
    X = mnist.data[:500]  # Limit to 500 instances for testing
    y = mnist.target[:500].astype(int)  # Ensure targets are integers

    # Process data and store in database
    X_embedded, y = data_processor.process_data(X, y)

    # Fetch and preprocess data for visualization
    query = text("SELECT * FROM vector_data")
    X_embedded, y = data_processor.fetch_and_preprocess_data(query)

    # Visualize data
    data_processor.visualize_data(X_embedded, y)

if __name__ == "__main__":
    main()
