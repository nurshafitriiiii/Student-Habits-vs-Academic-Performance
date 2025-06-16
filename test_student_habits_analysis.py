import unittest
import pandas as pd
import numpy as np
from student_habits_vs_academic_performance_ml import (
    load_and_clean_data,
    encode_features,
    scale_features,
    train_and_evaluate_models
)

class TestStudentHabitsPipeline(unittest.TestCase):

    def setUp(self):
        # Create dummy dataset similar to the real one
        self.df = pd.DataFrame({
            'student_id': ['S1', 'S2'],
            'gender': ['Male', 'Female'],
            'part_time_job': ['Yes', 'No'],
            'diet_quality': ['Good', 'Fair'],
            'parental_education_level': ['Bachelor', np.nan],
            'internet_quality': ['Good', 'Poor'],
            'extracurricular_participation': ['Yes', 'No'],
            'exam_score': [88, 72]
        })

    def test_load_and_clean_data(self):
        # Save sample to CSV to simulate loading
        self.df.to_csv('test_dummy.csv', index=False)
        df_clean = load_and_clean_data('test_dummy.csv')
        self.assertFalse(df_clean['parental_education_level'].isna().any(), "Missing values not filled")

    def test_encode_features(self):
        df_encoded = encode_features(self.df)
        expected_cols = {'dq_e', 'pel_e', 'iq_e', 'gender_Male', 'part_time_job_Yes', 'extracurricular_participation_Yes'}
        self.assertTrue(expected_cols.issubset(df_encoded.columns), "Encoding failed or incomplete")

    def test_scale_features(self):
        X = pd.DataFrame({
            'study_hours': [1, 2, 3],
            'sleep_hours': [6, 7, 8]
        })
        X_scaled = scale_features(X)
        self.assertTrue(np.allclose(X_scaled.mean(axis=0), 0, atol=1e-7), "Mean not 0 after scaling")
        self.assertTrue(np.allclose(X_scaled.std(axis=0), 1, atol=1e-7), "Std not 1 after scaling")

    def test_train_and_evaluate_models(self):
        X = np.random.rand(10, 3)
        y = np.random.rand(10)
        X_train, X_test = X[:8], X[8:]
        y_train, y_test = y[:8], y[8:]
        results = train_and_evaluate_models(X_train, X_test, y_train, y_test)
        self.assertIsNone(results, "Expected no return (prints only)")

if __name__ == "__main__":
    unittest.main()
