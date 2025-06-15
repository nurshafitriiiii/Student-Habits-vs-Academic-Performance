# -*- coding: utf-8 -*-
# Ensures the script supports UTF-8 encoding, useful for text data with special characters.

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os  # Used to create directories and handle file paths

# Scikit-learn modules for regression models, preprocessing, and evaluation
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

# --- Create a directory to save plots ---
# Ensures 'plots' folder exists to save visualizations
if not os.path.exists('plots'):
    os.makedirs('plots')

# --- Load the dataset ---
# Assumes the dataset is in the same folder
df = pd.read_csv('student_habits_performance.csv')
df = pd.DataFrame(df)  # Just to ensure it's a DataFrame
print("DataFrame Head:")
print(df.head())

print("\nDataFrame Info:")
df.info()  # Prints column types, non-null counts to understand the dataset structure

# --- Data Cleaning: Handle missing values ---
# Fill missing 'parental_education_level' with the most common (mode) value to maintain consistency
df['parental_education_level'] = df['parental_education_level'].fillna(df['parental_education_level'].mode()[0])
print("\nMissing Values Count After Filling:")
print(df.isna().sum())  # Verify no missing values remain

# --- Select categorical columns for visualization ---
cat_col = df.select_dtypes(include='object').columns.tolist()
cat_col.remove('student_id')  # Remove 'student_id' since it's just an identifier

# --- Plot 1: Categorical Feature Distributions ---
# Helps understand the frequency of categories (e.g., gender, part-time job, etc.)
plt.figure(figsize=(10, 7))
for i, col in enumerate(cat_col):
    plt.subplot(2, 3, i + 1)
    sns.countplot(data=df, x=col)
    plt.title(f'Distribution of {col}')
    plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig('plots/categorical_distributions.png', dpi=300, bbox_inches='tight')
plt.show()

# --- Plot 2: Numerical Feature Distributions ---
# Helps detect skewness, outliers, and data spread
num_col = df.select_dtypes(exclude='object').columns.tolist()
plt.figure(figsize=(12, 12))
for i, col in enumerate(num_col):
    plt.subplot(3, 3, i + 1)
    sns.histplot(df[col], bins=20, kde=True)
    plt.title(f'Distribution of {col}')
plt.tight_layout()
plt.savefig('plots/numerical_distributions.png', dpi=300, bbox_inches='tight')
plt.show()

# --- Drop identifier column before modeling ---
df2 = df.drop('student_id', axis=1)

# --- Encoding Categorical Variables ---
# Manually mapping ordinal categories to numerical values
diet_quality = {'Poor': 0, 'Fair': 1, 'Good': 2}
parental_education_level = {'High School': 0, 'Bachelor': 1, 'Master': 2}
internet_quality = {'Poor': 0, 'Average': 1, 'Good': 2}
df2['dq_e'] = df2['diet_quality'].map(diet_quality)
df2['pel_e'] = df2['parental_education_level'].map(parental_education_level)
df2['iq_e'] = df2['internet_quality'].map(internet_quality)

# One-hot encode nominal categories (no intrinsic order), like gender and job status
dummies = pd.get_dummies(df[['gender', 'part_time_job', 'extracurricular_participation']],
                         drop_first=True, dtype=int)

# Combine encoded and one-hot features
df3 = pd.concat([df2, dummies], axis=1)

# Drop original categorical columns now that they've been encoded
df3 = df3.drop(['gender', 'part_time_job', 'diet_quality', 'parental_education_level',
                'internet_quality', 'extracurricular_participation'], axis=1)

print("\nDataFrame Head After Encoding and Feature Engineering:")
print(df3.head())

# --- Plot 3: Correlation Heatmap ---
# Shows relationship strength between features and with exam_score
plt.figure(figsize=(15, 12))
corr = df3.corr()
sns.heatmap(corr, annot=True, fmt=".2f", cmap='coolwarm')
plt.title('Correlation Heatmap')
plt.savefig('plots/correlation_heatmap.png', dpi=300, bbox_inches='tight')
plt.show()

# --- Feature Scaling ---
# Standardize features to have mean = 0 and std = 1 (important for models like Lasso, Ridge)
X = df3.drop('exam_score', axis=1)
y = df3['exam_score']
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# --- Train-Test Split ---
# Split data into 80% training and 20% testing for unbiased model evaluation
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# --- Model Training and Evaluation ---
# Fit various regression models to compare performance

models = {
    "Linear Regression": LinearRegression(),               # Basic linear model
    "Ridge": Ridge(alpha=1.0),                             # Regularized linear regression to prevent overfitting
    "Lasso": Lasso(alpha=0.1),                             # L1-regularization, can reduce some feature weights to zero
    "Random Forest": RandomForestRegressor(n_estimators=100, max_depth=5, random_state=42),  # Ensemble of decision trees
    "Gradient Boosting": GradientBoostingRegressor(n_estimators=100, random_state=42)         # Boosting algorithm for better accuracy
}

print("\n--- Model Performance ---")
for name, model in models.items():
    model.fit(X_train, y_train)  # Train the model
    y_pred = model.predict(X_test)  # Predict on test data
    # Print R² score (explained variance) and RMSE (error magnitude)
    print(f"{name} → R²: {r2_score(y_test, y_pred):.3f}, RMSE: {np.sqrt(mean_squared_error(y_test, y_pred)):.2f}")
