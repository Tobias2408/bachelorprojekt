import pandas as pd
from sklearn.preprocessing import LabelEncoder, OneHotEncoder

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
    def fit_onehot_encoding(df, columns):
        encoders = {}
        onehot_encoded_cols = []
        for col in columns:
            ohe = OneHotEncoder(sparse_output=False, drop='first')
            transformed_data = ohe.fit_transform(df[[col]])
            encoded_df = pd.DataFrame(transformed_data, columns=[f"{col}_{category}" for category in ohe.categories_[0][1:]])
            encoded_df = encoded_df.astype(int)  # Convert to int
            df = pd.concat([df.drop(col, axis=1), encoded_df], axis=1)
            encoders[col] = ohe
            onehot_encoded_cols.extend(encoded_df.columns)
        return df, encoders, onehot_encoded_cols

    @staticmethod
    def transform_onehot_encoding(df, columns, encoders):
        for col in columns:
            if col in encoders:
                ohe = encoders[col]
                transformed_data = ohe.transform(df[[col]])
                encoded_df = pd.DataFrame(transformed_data, columns=[f"{col}_{category}" for category in ohe.categories_[0][1:]])
                encoded_df = encoded_df.astype(int)  # Convert to int
                df = pd.concat([df.drop(col, axis=1), encoded_df], axis=1)
            else:
                raise ValueError(f"OneHot encoder for column '{col}' not provided.")
        return df

    @staticmethod
    def fit_transform_onehot_encoding(df, columns):
        df, encoders, onehot_encoded_cols = CategoricalEncoder.fit_onehot_encoding(df, columns)
        return df, encoders, onehot_encoded_cols
