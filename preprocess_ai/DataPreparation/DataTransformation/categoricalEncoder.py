import openai
import unittest
import pandas as pd
from pandas.testing import assert_frame_equal
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
import numpy as np
import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Set OpenAI API key
openai.api_key = os.getenv('OPENAI_API_KEY')

class WordEmbedder:
    def __init__(self, model_name='text-embedding-ada-002'):
        self.model_name = model_name

    def get_embedding(self, text):
        response = openai.Embedding.create(input=[text], model=self.model_name)
        return response['data'][0]['embedding']

    def embed_word(self, word):
        return self.get_embedding(word)

    def embed_words(self, words):
        return {word: self.embed_word(word) for word in words}

    def embed_column(self, df, column):
        embeddings = df[column].apply(lambda x: self.embed_word(str(x)) if pd.notnull(x) else np.zeros(1536))
        embeddings_df = pd.DataFrame(embeddings.tolist(), index=df.index)
        return pd.concat([df.drop(columns=[column]), embeddings_df], axis=1)

class CategoricalEncoder:
    @staticmethod
    def fit_label_encoding(df, columns):
        encoders = {}
        for col in columns:
            le = LabelEncoder()
            df[col] = le.fit_transform(df[col])
            encoders[col] = le
        return df, encoders

    @staticmethod
    def transform_label_encoding(df, columns, encoders):
        for col in columns:
            if col in encoders:
                le = encoders[col]
                df[col] = le.transform(df[col])
            else:
                raise ValueError(f"Label encoder for column '{col}' not provided.")
        return df

    @staticmethod
    def fit_transform_label_encoding(df, columns):
        df, encoders = CategoricalEncoder.fit_label_encoding(df, columns)
        return df, encoders

    @staticmethod
    def fit_onehot_encoding(df, columns, max_categories=None):
        encoders = {}
        onehot_encoded_cols = []
        for col in columns:
            if df[col].dtype != 'object' or df[col].nunique() == 1:
                # Skip non-categorical or constant columns
                df.drop(columns=[col], inplace=True)
                continue
            
            if max_categories is not None and df[col].nunique() > max_categories:
                top_categories = df[col].value_counts().index[:max_categories]
                df[col] = df[col].apply(lambda x: x if x in top_categories else 'Other')
            
            ohe = OneHotEncoder(sparse_output=False, drop='first')
            transformed_data = ohe.fit_transform(df[[col]])
            encoded_df = pd.DataFrame(transformed_data, columns=[f"{col}_{category}" for category in ohe.categories_[0][1:]])
            encoded_df = encoded_df.astype(int)
            df = pd.concat([df.drop(col, axis=1), encoded_df], axis=1)
            encoders[col] = ohe
            onehot_encoded_cols.extend(encoded_df.columns)
            print(f"One-Hot Encoded {col}:")
            print(encoded_df.head())
        return df, encoders, onehot_encoded_cols

    @staticmethod
    def transform_onehot_encoding(df, columns, encoders):
        for col in columns:
            if col in encoders:
                ohe = encoders[col]
                transformed_data = ohe.transform(df[[col]])
                encoded_df = pd.DataFrame(transformed_data, columns=[f"{col}_{category}" for category in ohe.categories_[0][1:]])
                encoded_df = encoded_df.astype(int)
                df = pd.concat([df.drop(col, axis=1), encoded_df], axis=1)
                print(f"Transformed One-Hot Encoded {col}:")
                print(encoded_df.head())
            else:
                raise ValueError(f"OneHot encoder for column '{col}' not provided.")
        return df

    @staticmethod
    def fit_transform_onehot_encoding(df, columns, max_categories=None):
        df, encoders, onehot_encoded_cols = CategoricalEncoder.fit_onehot_encoding(df, columns, max_categories)
        return df, encoders, onehot_encoded_cols

# Example Usage
if __name__ == "__main__":
    # Example DataFrame
    data = {
        'text': ['hello', 'world', 'foo', 'bar', 'openai'],
        'category': ['a', 'b', 'a', 'b', 'a']
    }
    df = pd.DataFrame(data)

    # Word Embedding
    embedder = WordEmbedder()
    df = embedder.embed_column(df, 'text')
    print(df.head())

    # Categorical Encoding
    encoder = CategoricalEncoder()
    df, encoders, onehot_encoded_cols = encoder.fit_transform_onehot_encoding(df, ['category'])
    print(df.head())
