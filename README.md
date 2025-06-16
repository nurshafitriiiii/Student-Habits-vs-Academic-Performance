# Student Habits vs Academic Performance (Machine Learning Project)

## 1. Project Overview

This project investigates how various lifestyle and demographic factors influence student academic performance using machine learning techniques. The dataset contains information such as students' diet quality, internet access, parental education, part-time job status, extracurricular participation, and more, with the primary goal of predicting each student's **exam score**[1].

## 2. Objective

The objective of this project is to build and compare predictive models that estimate students' exam performance based on their personal habits and demographic characteristics[1].

## 3. Files Included

| File Name                          | Description                                                      |
|-------------------------------------|------------------------------------------------------------------|
| `student_habits_performance.csv`    | Source dataset (required for the code to run)[1]                 |
| `student_habits_vs_academic_perfomance.py` | Main Python script containing the data analysis and ML pipeline[1] |
| `README.md`                        | This file — explains the purpose, structure, and usage of the project[1] |

## 4. Technologies Used

- **Language:** Python 3[1]
- **Libraries:**  
  - `pandas`, `numpy` for data handling[1]
  - `matplotlib`, `seaborn` for visualizations[1]
  - `scikit-learn` for machine learning[1]

## 5. Features & Workflow

### 5.1 Data Loading & Cleaning

- Loads the CSV file into a DataFrame[1].
- Fills missing values in the `parental_education_level` column with the most common value to maintain data consistency[1].

### 5.2 Visualization

- Plots distributions for categorical features (e.g., gender, job status)[1].
- Plots distributions for numerical features (e.g., study hours)[1].
- Generates a correlation heatmap to visualize relationships between variables and the target (exam score)[1].

### 5.3 Feature Engineering

- Applies manual label encoding for ordinal features: `diet_quality`, `internet_quality`, and `parental_education_level`[1].
- Uses one-hot encoding for nominal categorical features: `gender`, `part_time_job`, and `extracurricular_participation`[1].

### 5.4 Data Scaling

- Standardizes numeric features using `StandardScaler` to ensure all features contribute equally to model training[1].

### 5.5 Modeling

- Splits the data into 80% training and 20% testing sets for unbiased evaluation[1].
- Trains five different regression models:
  - Linear Regression
  - Ridge Regression
  - Lasso Regression
  - Random Forest
  - Gradient Boosting[1]

### 5.6 Model Evaluation

- Evaluates models using R² Score (explained variance) and RMSE (Root Mean Squared Error)[1].
- Outputs the performance metrics for each model directly to the console[1].

## 6. Outputs

- **Performance metrics** for each model are printed in the terminal, for example:
  ```
  Linear Regression → R²: 0.XX, RMSE: XX.XX
  Ridge → R²: 0.XX, RMSE: XX.XX
  ...
  ```
- **Saved Plots** (in `/plots` folder):
  - `categorical_distributions.png`
  - `numerical_distributions.png`
  - `correlation_heatmap.png`[1]

## 7. How to Run

1. Place `student_habits_performance.csv` in the same directory as the `.py` file[1].
2. Run the script:
   ```bash
   python student_habits_vs_academic_perfomance.py
   ```
3. Check the `plots/` folder for saved charts[1].
4. Read the terminal output for model performance comparisons[1].

## 8. Notes

- The code supports UTF-8 characters for compatibility with diverse text data[1].
- The script is modular and well-commented for clarity and ease of understanding[1].
- Visualization files can be included in your final report or presentation for better illustration of findings[1].

## 9. Credit

This project and code are adapted from the original Kaggle notebook by Jayaant Anaath, available at:  
https://www.kaggle.com/code/jayaantanaath/student-habits-vs-academic-performance-ml-90/[1]

---

**Reference:**  
[1] https://www.kaggle.com/code/jayaantanaath/student-habits-vs-academic-performance-ml-90/

[1] education.technical_concepts
