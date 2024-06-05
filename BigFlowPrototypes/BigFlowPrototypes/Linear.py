import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from keras.models import Sequential
from keras.layers import Dense, Dropout
import time

# Timing function
def timeit(method):
    def timed(*args, **kwargs):
        start_time = time.time()
        result = method(*args, **kwargs)
        end_time = time.time()
        print(f"{method.__name__} took {end_time - start_time:.2f} seconds")
        return result
    return timed

# Define neural network model
def create_model(input_dim, optimizer='adam', dropout_rate=0.0, neurons=32):
    model = Sequential()
    model.add(Dense(neurons, input_dim=input_dim, activation='relu'))
    model.add(Dropout(dropout_rate))
    model.add(Dense(neurons, activation='relu'))
    model.add(Dropout(dropout_rate))
    model.add(Dense(1, activation='linear'))
    model.compile(loss='mean_squared_error', optimizer=optimizer)
    return model

@timeit
def process_data(input_file, label):
    print("Loading data...")
    df = pd.read_csv(input_file)

    # Identify non-numeric columns
    non_numeric_columns = [col for col in df.columns if df[col].dtype == 'object']

    # Drop non-numeric columns
    df.drop(columns=non_numeric_columns, inplace=True)

    # Handle missing values
    df = df.fillna(df.mean(numeric_only=True))

    # Splitting data
    X = df.drop(columns=[label])
    y = df[label]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Standardize the data
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    
    return X_train, X_test, y_train, y_test, X, y

@timeit
def train_and_evaluate_models(X_train, X_test, y_train, y_test, X, y):
    input_dim = X_train.shape[1]

    # Neural Network Model
    nn_model = create_model(input_dim)
    nn_model.fit(X_train, y_train, epochs=50, batch_size=10, verbose=1)
    nn_y_pred = nn_model.predict(X_test)
    nn_mse = mean_squared_error(y_test, nn_y_pred)
    nn_r2 = r2_score(y_test, nn_y_pred)

    # Linear Regression Model
    lr_model = LinearRegression()
    lr_model.fit(X_train, y_train)
    lr_y_pred = lr_model.predict(X_test)
    lr_mse = mean_squared_error(y_test, lr_y_pred)
    lr_r2 = r2_score(y_test, lr_y_pred)
    lr_cv_mse = -cross_val_score(lr_model, X, y, cv=3, scoring='neg_mean_squared_error').mean()

    # Ridge Regression Model
    ridge_model = Ridge()
    ridge_model.fit(X_train, y_train)
    ridge_y_pred = ridge_model.predict(X_test)
    ridge_mse = mean_squared_error(y_test, ridge_y_pred)
    ridge_r2 = r2_score(y_test, ridge_y_pred)
    ridge_cv_mse = -cross_val_score(ridge_model, X, y, cv=3, scoring='neg_mean_squared_error').mean()

    # Random Forest Regressor Model
    rf_model = RandomForestRegressor()
    rf_model.fit(X_train, y_train)
    rf_y_pred = rf_model.predict(X_test)
    rf_mse = mean_squared_error(y_test, rf_y_pred)
    rf_r2 = r2_score(y_test, rf_y_pred)
    rf_cv_mse = -cross_val_score(rf_model, X, y, cv=3, scoring='neg_mean_squared_error').mean()

    # Gradient Boosting Regressor Model
    gb_model = GradientBoostingRegressor()
    gb_model.fit(X_train, y_train)
    gb_y_pred = gb_model.predict(X_test)
    gb_mse = mean_squared_error(y_test, gb_y_pred)
    gb_r2 = r2_score(y_test, gb_y_pred)
    gb_cv_mse = -cross_val_score(gb_model, X, y, cv=3, scoring='neg_mean_squared_error').mean()

    # Display results
    print("\nNeural Network Model Performance:")
    print(f"Mean Squared Error: {nn_mse}")
    print(f"R^2 Score: {nn_r2}")

    print("\nLinear Regression Model Performance:")
    print(f"Mean Squared Error: {lr_mse}")
    print(f"R^2 Score: {lr_r2}")
    print(f"Cross-Validated MSE: {lr_cv_mse}")

    print("\nRidge Regression Model Performance:")
    print(f"Mean Squared Error: {ridge_mse}")
    print(f"R^2 Score: {ridge_r2}")
    print(f"Cross-Validated MSE: {ridge_cv_mse}")

    print("\nRandom Forest Regressor Model Performance:")
    print(f"Mean Squared Error: {rf_mse}")
    print(f"R^2 Score: {rf_r2}")
    print(f"Cross-Validated MSE: {rf_cv_mse}")

    print("\nGradient Boosting Regressor Model Performance:")
    print(f"Mean Squared Error: {gb_mse}")
    print(f"R^2 Score: {gb_r2}")
    print(f"Cross-Validated MSE: {gb_cv_mse}")

def main(input_file, label):
    X_train, X_test, y_train, y_test, X, y = process_data(input_file, label)
    train_and_evaluate_models(X_train, X_test, y_train, y_test, X, y)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process input data for AI models.")
    parser.add_argument('input_file', type=str, help='Path to the input data file (e.g., movies.csv)')
    parser.add_argument('label', type=str, help='Name of the label column (e.g., Popularity)')
    
    args = parser.parse_args()
    
    X_train, X_test, y_train, y_test = main(args.input_file, args.label)
    
    print("Data processing complete.")