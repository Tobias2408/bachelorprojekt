#!/usr/bin/env python
# coding: utf-8

# In[7]:





# In[25]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


# In[32]:


df = pd.read_csv('titanic.csv')


# In[33]:


# Display the first few rows of the dataset
print(df.head())

# Check the dimensions of the dataset
print(df.shape)

# Get summary statistics of numerical variables
print(df.describe())

# Check the data types of variables
print(df.dtypes)

# Check for missing values
print(df.isnull().sum())


# In[34]:


mean_values = df.select_dtypes(include=['number']).mean()

# Then, use fillna() on the DataFrame, passing the dictionary to fill each column with its respective mean
df = df.fillna(value=mean_values)


# In[35]:


print(df)


# In[36]:


def is_plotable(series):
    """ Check if the column is suitable for plotting. """
    if series.nunique() > 20:  # Adjust the threshold as needed
        return False
    return True

# Function to determine if a column is numeric
def is_numeric(series):
    return pd.api.types.is_numeric_dtype(series)

# Function to determine if a column is categorical with a reasonable number of unique values
def is_categorical(series):
    if pd.api.types.is_categorical_dtype(series) or series.dtype == object:
        return is_plotable(series)
    return False

# Plot each column in the DataFrame
for column in df.columns:
    # Skip columns with high cardinality or non-informative data
    if column in ['Id', 'Name', 'Original_name']:  # Add other columns to skip as necessary
        continue
    
    if is_numeric(df[column]):
        # Histogram for numeric data
        plt.figure(figsize=(10, 4))
        plt.hist(df[column].dropna(), bins=15, color='skyblue')
        plt.title(f'Distribution of {column}')
        plt.xlabel(column)
        plt.ylabel('Frequency')
        plt.show()

        # Box plot for numeric data
        plt.figure(figsize=(10, 4))
        sns.boxplot(x=df[column].dropna())
        plt.title(f'Box Plot of {column}')
        plt.xlabel(column)
        plt.show()

    elif is_categorical(df[column]):
        # Count plot for categorical data
        plt.figure(figsize=(10, 4))
        sns.countplot(x=column, data=df)
        plt.title(f'Count Plot of {column}')
        plt.xlabel(column)
        plt.ylabel('Count')
        plt.show()


# In[41]:


def plot_histograms(df, bins=10, figsize=(10, 5)):
    """
    Plot histograms for all numeric columns in the DataFrame.

    Args:
    df (DataFrame): The pandas DataFrame containing the data.
    bins (int, optional): Number of bins for the histogram. Default is 10.
    figsize (tuple, optional): Figure size, given as (width, height). Default is (10, 5).
    """
    # Identify numeric columns by checking if they can be coerced to numeric types
    numeric_columns = df.select_dtypes(include=['number']).columns

    # Create a histogram for each numeric column
    for column in numeric_columns:
        plt.figure(figsize=figsize)
        plt.hist(df[column].dropna(), bins=bins, color='skyblue', edgecolor='black')  # Drop NA values for plotting
        plt.title(f'Histogram of {column}')
        plt.xlabel(column)
        plt.ylabel('Frequency')
        plt.grid(True)
        plt.show()


# In[42]:


plot_histograms(df)


# In[39]:


# Assuming df is already loaded
# df = pd.read_csv('path/to/your/data.csv')

def is_numeric(series):
    """Check if a column is numeric."""
    return pd.api.types.is_numeric_dtype(series)

def is_categorical(series, threshold=10):
    """Check if a column is categorical. Threshold for unique counts can be adjusted."""
    return series.dtype == object and series.nunique() <= threshold

# Identify numeric and categorical columns
numeric_columns = [col for col in df.columns if is_numeric(df[col])]
categorical_columns = [col for col in df.columns if is_categorical(df[col])]

# Generic Correlation Analysis for numeric columns
if len(numeric_columns) > 1:
    correlation_matrix = df[numeric_columns].corr()
    print("Correlation matrix:\n", correlation_matrix)

    # Optionally, display a heatmap of the correlation matrix
    plt.figure(figsize=(8, 6))
    sns.heatmap(correlation_matrix, annot=True, fmt=".2f", cmap='coolwarm')
    plt.title('Correlation Matrix')
    plt.show()

# Generic Cross-Tabulation for pairs of categorical columns
for i in range(len(categorical_columns)):
    for j in range(i + 1, len(categorical_columns)):
        cross_tab = pd.crosstab(df[categorical_columns[i]], df[categorical_columns[j]])
        print(f"Cross-tabulation between {categorical_columns[i]} and {categorical_columns[j]}:\n", cross_tab)
        print("\n")


# In[ ]:




