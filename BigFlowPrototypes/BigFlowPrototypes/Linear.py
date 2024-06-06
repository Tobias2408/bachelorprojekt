import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
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

def main(input_file, label):
    return process_data(input_file, label)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process input data for AI models.")
    parser.add_argument('input_file', type=str, help='Path to the input data file (e.g., movies.csv)')
    parser.add_argument('label', type=str, help='Name of the label column (e.g., Popularity)')
    
    args = parser.parse_args()
    
    X_train, X_test, y_train, y_test, X, y = main(args.input_file, args.label)
    
    print("Data processing complete.")
