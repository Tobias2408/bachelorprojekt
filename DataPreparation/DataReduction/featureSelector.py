import pandas as pd
from sklearn.feature_selection import SelectKBest, f_classif, f_regression, RFE
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LogisticRegression
from sklearn.base import is_classifier, is_regressor

class FeatureSelector:
    
    @staticmethod
    def select_k_best(df, target, k=10, score_func=f_classif):
        """
        Selects the k best features based on a statistical test.
        
        Parameters:
        df (pd.DataFrame): DataFrame containing the data.
        target (pd.Series): Target variable.
        k (int): Number of top features to select.
        score_func (callable): Function to compute the score between features and target.
        
        Returns:
        pd.DataFrame: DataFrame with the selected features.
        """
        selector = SelectKBest(score_func=score_func, k=k)
        selector.fit(df, target)
        selected_columns = df.columns[selector.get_support()]
        return df[selected_columns]

    @staticmethod
    def select_by_model(df, target, model=None, n_features_to_select=10):
        """
        Selects features based on a model's feature importance.
        
        Parameters:
        df (pd.DataFrame): DataFrame containing the data.
        target (pd.Series): Target variable.
        model: Model to use for feature importance. Default is RandomForestClassifier.
        n_features_to_select (int): Number of top features to select.
        
        Returns:
        pd.DataFrame: DataFrame with the selected features.
        """
        if model is None:
            if is_classifier(target):
                model = RandomForestClassifier(n_estimators=100, random_state=42)
            else:
                model = RandomForestRegressor(n_estimators=100, random_state=42)
                
        model.fit(df, target)
        feature_importances = pd.Series(model.feature_importances_, index=df.columns)
        selected_columns = feature_importances.nlargest(n_features_to_select).index
        return df[selected_columns]

    @staticmethod
    def select_by_rfe(df, target, model=None, n_features_to_select=10):
        """
        Selects features using Recursive Feature Elimination (RFE).
        
        Parameters:
        df (pd.DataFrame): DataFrame containing the data.
        target (pd.Series): Target variable.
        model: Model to use for RFE. Default is LogisticRegression.
        n_features_to_select (int): Number of top features to select.
        
        Returns:
        pd.DataFrame: DataFrame with the selected features.
        """
        if model is None:
            model = LogisticRegression(solver='liblinear', random_state=42)
        
        rfe = RFE(model, n_features_to_select=n_features_to_select)
        rfe.fit(df, target)
        selected_columns = df.columns[rfe.support_]
        return df[selected_columns]

