import numpy as np

class SimpleScaler:
    def standardize(self, data):
        """
        Standardize the data to have zero mean and unit variance.
        Uses sample standard deviation (N-1 in the denominator).
        """
        mean = np.mean(data, axis=0)
        std = np.std(data, axis=0, ddof=1)  # Calculate the sample standard deviation
        return (data - mean) / std, mean, std

    def normalize(self, data):
        """
        Normalize the data to have a range of [0, 1].
        """
        min_val = np.min(data, axis=0)
        max_val = np.max(data, axis=0)
        return (data - min_val) / (max_val - min_val), min_val, max_val
