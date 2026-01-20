import os
import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

def prepare_features(schedule_df):
    """
    Convert raw schedule DataFrame into features for ML.
    """
    df = schedule_df.copy()

    # Reset index if it's multi-index to flatten it
    if isinstance(df.index, pd.MultiIndex):
        df = df.reset_index()

    # Standardize column names
    # soccerdata typically uses 'home_team', 'away_team', 'home_score', 'away_score'
    # But sometimes 'home_goals', 'away_goals' might be present from other sources.
    if 'home_score' in df.columns and 'home_goals' not in df.columns:
        df.rename(columns={'home_score': 'home_goals', 'away_score': 'away_goals'}, inplace=True)

    # Drop rows where scores are NaN (future matches) for training
    df_train = df.dropna(subset=['home_goals', 'away_goals']).copy()

    # Create target: 0=Away, 1=Draw, 2=Home
    conditions = [
        (df_train['home_goals'] > df_train['away_goals']),
        (df_train['home_goals'] == df_train['away_goals']),
        (df_train['home_goals'] < df_train['away_goals'])
    ]
    choices = [2, 1, 0] # Home Win, Draw, Away Win
    df_train['target'] = np.select(conditions, choices)

    # Encode teams
    le = LabelEncoder()
    # Fit on all unique teams in the full dataframe (including future matches)
    all_teams = pd.concat([df['home_team'], df['away_team']]).unique()
    le.fit(all_teams)

    df_train['home_code'] = le.transform(df_train['home_team'])
    df_train['away_code'] = le.transform(df_train['away_team'])

    # --- Feature Engineering: Recent Form ---
    # We need to calculate this carefully so we don't leak future data.
    # We will calculate rolling stats based on the date.

    # Ensure date is datetime
    if 'date' in df_train.columns:
        df_train['date'] = pd.to_datetime(df_train['date'])
        df_train = df_train.sort_values('date')

    # Calculate simple rolling averages for goals and points
    # This is a simplified approach. A more robust one iterates through matches.

    stats = {} # team -> list of match stats

    home_form_points = []
    away_form_points = []
    home_avg_goals = []
    away_avg_goals = []

    # Initialize stats for all teams
    for team in all_teams:
        stats[team] = {
            'points': [],
            'goals_for': [],
            'goals_against': []
        }

    for idx, row in df_train.iterrows():
        ht = row['home_team']
        at = row['away_team']
        hg = row['home_goals']
        ag = row['away_goals']

        # Get pre-match stats (last 5 games)
        def get_avg(team, metric):
            l = stats[team][metric]
            return np.mean(l[-5:]) if len(l) > 0 else 0.0

        def get_form(team):
            l = stats[team]['points']
            return sum(l[-5:]) if len(l) > 0 else 0

        home_form_points.append(get_form(ht))
        away_form_points.append(get_form(at))
        home_avg_goals.append(get_avg(ht, 'goals_for'))
        away_avg_goals.append(get_avg(at, 'goals_for'))

        # Update stats AFTER the match (for future matches)
        if hg > ag:
            hp, ap = 3, 0
        elif hg == ag:
            hp, ap = 1, 1
        else:
            hp, ap = 0, 3

        stats[ht]['points'].append(hp)
        stats[ht]['goals_for'].append(hg)
        stats[ht]['goals_against'].append(ag)

        stats[at]['points'].append(ap)
        stats[at]['goals_for'].append(ag)
        stats[at]['goals_against'].append(hg)

    df_train['home_form'] = home_form_points
    df_train['away_form'] = away_form_points
    df_train['home_goals_avg'] = home_avg_goals
    df_train['away_goals_avg'] = away_avg_goals

    feature_cols = ['home_code', 'away_code', 'home_form', 'away_form', 'home_goals_avg', 'away_goals_avg']

    return df_train, feature_cols, le, stats

def train_model(schedule_df, out_path="./model.pkl"):
    print(f"Training on {len(schedule_df)} matches...")
    df_train, feature_cols, le, current_stats = prepare_features(schedule_df)

    X = df_train[feature_cols]
    y = df_train['target']

    # Split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Train
    clf = RandomForestClassifier(n_estimators=100, random_state=42, max_depth=10)
    clf.fit(X_train, y_train)

    # Evaluate
    preds = clf.predict(X_test)
    acc = accuracy_score(y_test, preds)
    print(f"Model Accuracy: {acc:.2%}")
    print(classification_report(y_test, preds))

    # Save everything needed for prediction
    model_data = {
        "model": clf,
        "encoder": le,
        "stats": current_stats, # Saved to calculate features for next match
        "feature_cols": feature_cols,
        "accuracy": acc
    }

    joblib.dump(model_data, out_path)
    print(f"Model saved to {out_path}")
    return out_path

def load_model(path="./model.pkl"):
    if not os.path.exists(path):
        raise FileNotFoundError(path)
    return joblib.load(path)
