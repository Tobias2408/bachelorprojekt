import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tensorflow.keras.applications import VGG16
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import ast
import time

# Timing function decorator
def timeit(method):
    def timed(*args, **kwargs):
        start_time = time.time()
        result = method(*args, **kwargs)
        end_time = time.time()
        print(f"{method.__name__} took {end_time - start_time:.2f} seconds")
        return result
    return timed

class DataProcessor:
    def __init__(self, db_handler, model_name='VGG16', preprocess_func=None, image_size=(32, 32)):
        self.db_handler = db_handler
        self.model_name = model_name
        self.preprocess_func = preprocess_func
        self.image_size = image_size

    @timeit
    def process_data(self, X, y):
        # Reshape and preprocess the images for the pre-trained model
        print("Reshaping and preprocessing images...")
        X_images = X.values.reshape(-1, 28, 28, 1)
        X_images = np.repeat(X_images, 3, axis=-1)  # Convert to 3 channels
        X_images = np.array([np.pad(image, ((2, 2), (2, 2), (0, 0)), 'constant') for image in X_images])  # Resize to 32x32
        X_images = self.preprocess_func(X_images)

        # Load the pre-trained model and extract features
        print("Loading pre-trained model and extracting features...")
        if self.model_name == 'VGG16':
            model = VGG16(weights='imagenet', include_top=False, input_shape=self.image_size + (3,))
        else:
            raise ValueError(f"Model {self.model_name} not supported.")
        features = model.predict(X_images)
        X_embedded = features.reshape(X_images.shape[0], -1)
        print(f"Extracted features shape: {X_embedded.shape}")

        # Store vectors in database
        print("Storing vectors in PostgreSQL...")
        self.db_handler.store_vectors(X_embedded, y)
        return X_embedded, y

    def fetch_and_preprocess_data(self, query):
        df = self.db_handler.fetch_data(query)
        # Convert the 'features' column from string to list of floats
        df['features'] = df['features'].apply(lambda x: np.array(ast.literal_eval(x), dtype=float))
        X_embedded = np.array(df['features'].tolist())
        y = df['label'].astype(int)
        return X_embedded, y

    def visualize_data(self, X_embedded, y):
        print("Standardizing features...")
        scaler = StandardScaler()
        X_embedded = scaler.fit_transform(X_embedded)

        # Dimensionality reduction using PCA
        print("Reducing dimensionality to 3D for visualization...")
        pca_3d = PCA(n_components=3)
        X_pca_3d = pca_3d.fit_transform(X_embedded)

        # Creating a DataFrame for plotting
        plot_df = pd.DataFrame(X_pca_3d, columns=['PCA1', 'PCA2', 'PCA3'])
        plot_df['label'] = y

        # Plotting the 3D scatter plot using plotly
        import plotly.express as px
        fig = px.scatter_3d(plot_df, x='PCA1', y='PCA2', z='PCA3', color='label',
                            title='3D PCA of MNIST Vectors',
                            labels={'label': 'Digit'},
                            color_continuous_scale=px.colors.sequential.Viridis)
        fig.update_traces(marker=dict(size=3), selector=dict(mode='markers'))
        fig.show()
