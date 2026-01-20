import sys
import os
import pandas as pd
import numpy as np
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from model import prepare_features, train_model, load_model
from predict import predict_match

class TestMLPipeline(unittest.TestCase):
    def test_prepare_features(self):
        # Create dummy data
        data = {
            'home_team': ['A', 'B', 'A', 'C', 'B'],
            'away_team': ['B', 'A', 'C', 'A', 'C'],
            'home_goals': [2, 1, 3, 0, 1],
            'away_goals': [1, 1, 0, 2, 1],
            'date': pd.to_datetime(['2023-01-01', '2023-01-08', '2023-01-15', '2023-01-22', '2023-01-29'])
        }
        df = pd.DataFrame(data)

        df_train, feature_cols, le, stats = prepare_features(df)

        self.assertIn('home_form', df_train.columns)
        self.assertIn('home_code', df_train.columns)
        self.assertEqual(len(df_train), 5)

        # Check stats existence
        self.assertIn('A', stats)

    def test_predict_integration(self):
        # Assumes model.pkl exists from the pipeline run
        if not os.path.exists("./model.pkl"):
            print("Skipping integration test: model.pkl not found")
            return

        res = predict_match("Arsenal", "Liverpool", model_path="./model.pkl")
        self.assertIn("predicted_score", res)
        self.assertIn("probability", res)
        self.assertIn("home_exp", res)
        # Check that it didn't return an error dict
        self.assertNotIn("error", res)

if __name__ == "__main__":
    unittest.main()
