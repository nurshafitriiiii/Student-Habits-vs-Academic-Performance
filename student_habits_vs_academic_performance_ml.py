# -*- coding: utf-8 -*-
# Import necessary libraries for data manipulation, visualization, and machine learning
import os  # For directory operations like creating folders
import pandas as pd  # For data handling and analysis
import numpy as np  # For numerical operations
import matplotlib.pyplot as plt  # For plotting graphs
import seaborn as sns  # For enhanced data visualization

# Import regression models and utilities from scikit-learn
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.preprocessing import StandardScaler  # For feature scaling
from sklearn.model_selection import train_test_split  # For splitting dataset into train/test
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error  # For model evaluation

def create_plots_dir():
    """
    Create a directory named 'plots' if it does not exist.
    This folder will store all generated plot images.
    """
    if not os.path.exists('plots'):
        os.makedirs('plots')

def load_and_clean_data(filepath):
    """
    Load dataset from a CSV file and clean it by filling missing values.

    Parameters:
    - filepath (str): Path to the CSV file.

    Returns:
    - df (DataFrame): Cleaned pandas DataFrame.
    """
    df = pd.read_csv(filepath)  # Load data into DataFrame
    # Fill missing values in 'parental_education_level' column with the most frequent value (mode)
    df['parental_education_level'] = df['parental_education_level'].fillna(df['parental_education_level'].mode()[0])
    return df

def plot_categorical_distributions(df):
    """
    Plot count distributions for all categorical columns except 'student_id'.

    Parameters:
    - df (DataFrame): The input dataset.
    """
    # Select columns with data type 'object' (categorical)
    cat_cols = df.select_dtypes(include='object').columns.tolist()
    # Remove 'student_id' since it is an identifier, not a feature
    if 'student_id' in cat_cols:
        cat_cols.remove('student_id')
    plt.figure(figsize=(12, 8))  # Set figure size
    # Plot countplot for each categorical column in a grid of subplots
    for i, col in enumerate(cat_cols):
        plt.subplot(2, 3, i + 1)
        sns.countplot(data=df, x=col)
        plt.title(f'Distribution of {col}')
        plt.xticks(rotation=45)  # Rotate x-axis labels for readability
    plt.tight_layout()  # Adjust subplot spacing
    plt.savefig('plots/categorical_distributions.png', dpi=300, bbox_inches='tight')  # Save figure
    plt.show()  # Display plot

def plot_numerical_distributions(df):
    """
    Plot histograms with KDE for all numerical columns.

    Parameters:
    - df (DataFrame): The input dataset.
    """
    # Select columns that are not of type 'object' (numerical)
    num_cols = df.select_dtypes(exclude='object').columns.tolist()
    plt.figure(figsize=(12, 12))  # Set figure size
    # Plot histogram + KDE for each numerical column
    for i, col in enumerate(num_cols):
        plt.subplot(3, 3, i + 1)
        sns.histplot(df[col], bins=20, kde=True)
        plt.title(f'Distribution of {col}')
    plt.tight_layout()  # Adjust subplot spacing
    plt.savefig('plots/numerical_distributions.png', dpi=300, bbox_inches='tight')  # Save figure
    plt.show()  # Display plot

def encode_features(df):
    """
    Encode categorical features into numerical values suitable for modeling.

    Parameters:
    - df (DataFrame): Original dataset.

    Returns:
    - df3 (DataFrame): Dataset with encoded features.
    """
    df2 = df.drop('student_id', axis=1)  # Drop identifier column before encoding

    # Ordinal encoding: map ordered categories to integers
    diet_quality_map = {'Poor': 0, 'Fair': 1, 'Good': 2}
    parental_education_map = {'High School': 0, 'Bachelor': 1, 'Master': 2}
    internet_quality_map = {'Poor': 0, 'Average': 1, 'Good': 2}

    # Apply mappings to create new encoded columns
    df2['dq_e'] = df2['diet_quality'].map(diet_quality_map)
    df2['pel_e'] = df2['parental_education_level'].map(parental_education_map)
    df2['iq_e'] = df2['internet_quality'].map(internet_quality_map)

    # One-hot encode nominal categorical variables (no intrinsic order)
    dummies = pd.get_dummies(df2[['gender', 'part_time_job', 'extracurricular_participation']],
                             drop_first=True, dtype=int)
    # Concatenate one-hot encoded columns with the rest of the dataset
    df3 = pd.concat([df2, dummies], axis=1)

    # Drop original categorical columns after encoding to avoid redundancy
    df3 = df3.drop(['gender', 'part_time_job', 'diet_quality', 'parental_education_level',
                    'internet_quality', 'extracurricular_participation'], axis=1)
    return df3

def plot_correlation_heatmap(df):
    """
    Plot a heatmap showing correlations between features.

    Parameters:
    - df (DataFrame): Dataset with numerical features.
    """
    plt.figure(figsize=(15, 12))  # Set figure size
    corr = df.corr()  # Calculate correlation matrix
    sns.heatmap(corr, annot=True, fmt=".2f", cmap='coolwarm')  # Plot heatmap with annotations
    plt.title('Correlation Heatmap')
    plt.savefig('plots/correlation_heatmap.png', dpi=300, bbox_inches='tight')  # Save figure
    plt.show()  # Display plot

def scale_features(X):
    """
    Standardize features by removing the mean and scaling to unit variance.

    Parameters:
    - X (DataFrame or ndarray): Feature matrix.

    Returns:
    - X_scaled (ndarray): Scaled feature matrix.
    """
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)  # Fit scaler on data and transform
    return X_scaled

def train_and_evaluate_models(X_train, X_test, y_train, y_test):
    """
    Train multiple regression models and evaluate their performance.

    Parameters:
    - X_train, X_test (arrays): Training and testing feature sets.
    - y_train, y_test (arrays): Training and testing target values.

    Prints:
    - R², RMSE, and MAE for each model.
    """
    # Define a dictionary of models to train and evaluate
    models = {
        "Linear Regression": LinearRegression(),
        "Ridge": Ridge(alpha=1.0),
        "Lasso": Lasso(alpha=0.1),
        "Random Forest": RandomForestRegressor(n_estimators=100, max_depth=5, random_state=42),
        "Gradient Boosting": GradientBoostingRegressor(n_estimators=100, random_state=42)
    }

    print("\n--- Model Performance ---")
    # Train each model, predict, and print evaluation metrics
    for name, model in models.items():
        model.fit(X_train, y_train)  # Train model
        y_pred = model.predict(X_test)  # Predict on test data
        r2 = r2_score(y_test, y_pred)  # Coefficient of determination
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))  # Root mean squared error
        mae = mean_absolute_error(y_test, y_pred)  # Mean absolute error
        print(f"{name} → R²: {r2:.3f}, RMSE: {rmse:.2f}, MAE: {mae:.2f}")

def main():
    """
    Main function to run the full analysis pipeline:
    - Create plots directory
    - Load and clean data
    - Visualize distributions
    - Encode features
    - Plot correlation heatmap
    - Scale features
    - Split data
    - Train and evaluate models
    """
    create_plots_dir()  # Ensure 'plots' folder exists

    df = load_and_clean_data('student_habits_performance.csv')  # Load dataset

    print("DataFrame Head:")
    print(df.head())  # Preview first rows
    print("\nDataFrame Info:")
    df.info()  # Summary of dataset

    plot_categorical_distributions(df)  # Visualize categorical data
    plot_numerical_distributions(df)  # Visualize numerical data

    df_encoded = encode_features(df)  # Encode categorical variables
    print("\nDataFrame Head After Encoding and Feature Engineering:")
    print(df_encoded.head())  # Preview encoded data

    plot_correlation_heatmap(df_encoded)  # Visualize correlations

    # Separate features and target variable
    X = df_encoded.drop('exam_score', axis=1)
    y = df_encoded['exam_score']

    X_scaled = scale_features(X)  # Scale features for modeling

    # Split dataset into training and testing sets (80% train, 20% test)
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

    train_and_evaluate_models(X_train, X_test, y_train, y_test)  # Train models and evaluate

# Run main function if script is executed directly
if __name__ == "__main__":
    main()
