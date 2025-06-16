
# 📘 README: Student Habits vs Academic Performance (Machine Learning Project)

## 🔍 Project Overview
This project explores how various lifestyle factors influence student academic performance using machine learning. The dataset includes information on students' diet quality, internet access, parental education, part-time job status, extracurricular participation, and more — with the target variable being their **exam score**.

## 🎯 Objective
To build predictive models that estimate students' exam performance based on their personal habits and demographic characteristics.

## 📁 Files Included
| File Name | Description |
|-----------|-------------|
| `student_habits_performance.csv` | Source dataset (required for the code to run) |
| `student_habits_analysis.py` | Main Python script containing the full data analysis and machine learning pipeline |
| `plots/` | Folder containing all visualization output (.png files) |
| `README.md` | This file — explains the purpose, structure, and usage of the project |

## 🧪 Technologies Used
- **Language**: Python 3
- **Libraries**: 
  - `pandas`, `numpy` for data handling
  - `matplotlib`, `seaborn` for visualizations
  - `scikit-learn` for machine learning

## 🔧 Features & Workflow

1. **Data Loading & Cleaning**
   - Loads CSV file
   - Fills missing values in `parental_education_level`

2. **Visualization**
   - Categorical feature distributions
   - Numerical distributions
   - Correlation heatmap

3. **Feature Engineering**
   - Manual label encoding for ordinal features (`diet_quality`, `internet_quality`, `parental_education_level`)
   - One-hot encoding for nominal categorical features (`gender`, `part_time_job`, `extracurricular_participation`)

4. **Data Scaling**
   - StandardScaler used to normalize numeric features

5. **Modeling**
   - Splits data (80% training, 20% test)
   - Trains 5 models:
     - Linear Regression
     - Ridge Regression
     - Lasso Regression
     - Random Forest
     - Gradient Boosting

6. **Model Evaluation**
   - Uses R² Score and RMSE for comparison
   - Output printed directly to console

## 📊 Outputs
- **Performance metrics** for each model printed in the terminal:
  ```
  Linear Regression → R²: 0.XX, RMSE: XX.XX
  Ridge → R²: 0.XX, RMSE: XX.XX
  ...
  ```
- **Saved Plots** (in `/plots` folder):
  - `categorical_distributions.png`
  - `numerical_distributions.png`
  - `correlation_heatmap.png`

## ▶️ How to Run

1. Place `student_habits_performance.csv` in the same directory as the `.py` file.
2. Run the script:
   ```bash
   python student_habits_analysis.py
   ```
3. Check the `plots/` folder for saved charts.
4. Read the terminal output for model comparisons.

## 📌 Notes
- The code supports UTF-8 characters.
- Script is modular and well-commented for ease of understanding.
- Visualization files can be used in your final report or presentation.
