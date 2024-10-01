# data_exploration.py
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import missingno as msno

# Load dataset
def load_data(file_path):
    """Load the dataset from a CSV file."""
    df = pd.read_csv(file_path)
    return df

# Basic Data Inspection
def basic_inspection(df):
    """Perform basic inspection of the dataset."""
    print("First few rows of the dataset:")
    print(df.head(), "\n")

    print("Basic Information:")
    print(df.info(), "\n")

    print("Summary Statistics:")
    print(df.describe(include='all'), "\n")

    print("Missing Values:")
    print(df.isnull().sum(), "\n")

# Univariate Analysis
def univariate_analysis(df, numeric_cols, categorical_cols):
    """Perform univariate analysis on numeric and categorical columns."""
    
    # Numeric columns
    for col in numeric_cols:
        plt.figure(figsize=(14, 6))
        
        # Distribution
        plt.subplot(1, 2, 1)
        sns.histplot(df[col], bins=30, kde=True)
        plt.title(f'Distribution of {col}')
        plt.xlabel(col)
        plt.ylabel('Frequency')
        
        # Boxplot
        plt.subplot(1, 2, 2)
        sns.boxplot(x=df[col])
        plt.title(f'Boxplot of {col}')
        plt.xlabel(col)
        
        plt.tight_layout()
        plt.show()
    
    # Categorical columns
    for col in categorical_cols:
        plt.figure(figsize=(10, 6))
        sns.countplot(x=df[col])
        plt.title(f'Count Plot of {col}')
        plt.xlabel(col)
        plt.ylabel('Count')
        plt.show()

# Bivariate Analysis
def bivariate_analysis(df, numeric_cols, categorical_cols):
    """Perform bivariate analysis on numeric and categorical columns."""
    
    # Scatter plots for numeric pairs
    for i in range(len(numeric_cols)):
        for j in range(i + 1, len(numeric_cols)):
            plt.figure(figsize=(10, 6))
            sns.scatterplot(x=df[numeric_cols[i]], y=df[numeric_cols[j]])
            plt.title(f'Scatter Plot between {numeric_cols[i]} and {numeric_cols[j]}')
            plt.xlabel(numeric_cols[i])
            plt.ylabel(numeric_cols[j])
            plt.show()
    
    # Correlation heatmap
    plt.figure(figsize=(12, 8))
    correlation_matrix = df.corr()
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt='.2f')
    plt.title('Correlation Heatmap')
    plt.show()

    # Pairplot for multivariate analysis
    if len(numeric_cols) > 1:
        sns.pairplot(df, vars=numeric_cols)
        plt.title('Pairplot of Numeric Features')
        plt.show()

    # Categorical vs. Numeric
    for cat_col in categorical_cols:
        for num_col in numeric_cols:
            plt.figure(figsize=(10, 6))
            sns.boxplot(x=df[cat_col], y=df[num_col])
            plt.title(f'Boxplot of {num_col} by {cat_col}')
            plt.xlabel(cat_col)
            plt.ylabel(num_col)
            plt.show()

# Handling Missing Values
def handle_missing_values(df):
    """Visualize and handle missing values in the dataset."""
    
    plt.figure(figsize=(12, 8))
    msno.matrix(df)
    plt.title('Missing Values Matrix')
    plt.show()

    plt.figure(figsize=(12, 8))
    msno.heatmap(df)
    plt.title('Missing Values Heatmap')
    plt.show()

# Outlier Detection
def detect_outliers(df, numeric_cols):
    """Detect and visualize outliers in numeric columns."""
    for col in numeric_cols:
        plt.figure(figsize=(10, 6))
        sns.boxplot(x=df[col])
        plt.title(f'Boxplot of {col} (Outlier Detection)')
        plt.xlabel(col)
        plt.show()

        # Identify outliers
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        outliers = df[(df[col] < (Q1 - 1.5 * IQR)) | (df[col] > (Q3 + 1.5 * IQR))]
        print(f"Number of outliers in {col}: {len(outliers)}")

# Main function to perform all analyses
def main(file_path):
    df = load_data(file_path)

    # Define numeric and categorical columns (update based on your dataset)
    numeric_cols = ['numeric_column1', 'numeric_column2']  # Replace with actual numeric columns
    categorical_cols = ['categorical_column1', 'categorical_column2']  # Replace with actual categorical columns
    
    # Basic Inspection
    basic_inspection(df)
    
    # Univariate Analysis
    univariate_analysis(df, numeric_cols, categorical_cols)
    
    # Bivariate Analysis
    bivariate_analysis(df, numeric_cols, categorical_cols)
    
    # Handle Missing Values
    handle_missing_values(df)
    
    # Detect Outliers
    detect_outliers(df, numeric_cols)

# Explore Transactions
file_path = '../DataSets/AurgementedDataSet/bank.xlsx' 
main(file_path)

