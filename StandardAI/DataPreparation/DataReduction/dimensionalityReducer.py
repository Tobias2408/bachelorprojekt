import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

class DimensionalityReducer:
    
    @staticmethod
    def fit_pca(df, n_components, scale_data=True):
        """
        Fits a PCA model to the DataFrame.
        
        Parameters:
        df (pd.DataFrame): DataFrame containing the data.
        n_components (int or float): Number of components to keep.
        scale_data (bool): Whether to scale the data before applying PCA.
        
        Returns:
        PCA, StandardScaler: Fitted PCA model and scaler (if scaling was applied).
        """
        if scale_data:
            scaler = StandardScaler()
            df_scaled = scaler.fit_transform(df)
        else:
            scaler = None
            df_scaled = df
        
        pca = PCA(n_components=n_components)
        pca.fit(df_scaled)
        
        return pca, scaler

    @staticmethod
    def transform_pca(df, pca, scaler=None):
        """
        Transforms the DataFrame using the fitted PCA model.
        
        Parameters:
        df (pd.DataFrame): DataFrame containing the data.
        pca (PCA): Fitted PCA model.
        scaler (StandardScaler, optional): Fitted scaler.
        
        Returns:
        pd.DataFrame: DataFrame with reduced dimensions.
        """
        if scaler:
            df_scaled = scaler.transform(df)
        else:
            df_scaled = df
        
        df_pca = pca.transform(df_scaled)
        pca_columns = [f'PC{i+1}' for i in range(df_pca.shape[1])]
        df_pca = pd.DataFrame(df_pca, columns=pca_columns)
        
        return df_pca

    @staticmethod
    def fit_transform_pca(df, n_components, scale_data=True):
        """
        Fits a PCA model and transforms the DataFrame.
        
        Parameters:
        df (pd.DataFrame): DataFrame containing the data.
        n_components (int or float): Number of components to keep.
        scale_data (bool): Whether to scale the data before applying PCA.
        
        Returns:
        pd.DataFrame, PCA, StandardScaler: DataFrame with reduced dimensions, fitted PCA model, and scaler (if scaling was applied).
        """
        pca, scaler = DimensionalityReducer.fit_pca(df, n_components, scale_data)
        df_pca = DimensionalityReducer.transform_pca(df, pca, scaler)
        
        return df_pca, pca, scaler
