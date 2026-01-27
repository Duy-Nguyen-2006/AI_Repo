import os
import joblib
import pickle
import numpy as np
import pandas as pd
from scipy.stats import poisson
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import LabelEncoder
from functools import lru_cache

def train_baseline(schedule_df, out_path="./model.pkl"):
    # Simple home/away attack-defense rates from historical matches in schedule_df
    # Expects schedule_df with columns: home_team, away_team, home_goals, away_goals

    # Vectorized optimization:
    # 1. Convert columns to numeric, filling NaN with 0
    # 2. GroupBy home/away to aggregate stats
    # 3. Combine results

    # Work on a copy to avoid mutating the input DataFrame
    df_work = schedule_df.copy()

    # Ensure goals are numeric
    df_work["home_goals"] = pd.to_numeric(df_work["home_goals"], errors="coerce").fillna(0).astype(int)
    df_work["away_goals"] = pd.to_numeric(df_work["away_goals"], errors="coerce").fillna(0).astype(int)

    # Home stats: goals scored (for), goals conceded (against), matches played
    home_stats = df_work.groupby("home_team").agg(
        goals_for=("home_goals", "sum"),
        goals_against=("away_goals", "sum"),
        played=("home_team", "count")
    )

    # Away stats: goals scored (for), goals conceded (against), matches played
    away_stats = df_work.groupby("away_team").agg(
        goals_for=("away_goals", "sum"),
        goals_against=("home_goals", "sum"),
        played=("away_team", "count")
    )

    # Combine home and away stats
    # We use add(..., fill_value=0) to handle teams that might have only played home or away
    total_stats = home_stats.add(away_stats, fill_value=0)

    # Rename columns to match expected output
    # The previous loop created keys "for", "against", "played"
    total_stats = total_stats.rename(columns={"goals_for": "for", "goals_against": "against"})

    df = total_stats

    # Avoid division by zero
    df["avg_for"] = df["for"] / df["played"].replace(0, 1)
    df["avg_against"] = df["against"] / df["played"].replace(0, 1)

    # global average
    gavg = df["avg_for"].mean()

    # Convert to dictionary with "teams" key
    # The previous code used orient="index" which creates a dict of dicts: {team: {for: ..., against: ...}}
    model = {"teams": df.to_dict(orient="index"), "global_avg": float(gavg)}

    with open(out_path, "wb") as f:
        pickle.dump(model, f)
    return out_path

def prepare_features(schedule_df):
    df_train = schedule_df.copy()

    # Ensure numeric columns
    df_train["home_goals"] = pd.to_numeric(df_train["home_goals"], errors="coerce").fillna(0).astype(int)
    df_train["away_goals"] = pd.to_numeric(df_train["away_goals"], errors="coerce").fillna(0).astype(int)

    # Ensure date is datetime
    if "date" in df_train.columns:
        df_train["date"] = pd.to_datetime(df_train["date"])
    else:
        # If no date, assume index is temporal order
        df_train["date"] = df_train.index

    # Create target: 0=Away, 1=Draw, 2=Home
    conditions = [
        (df_train['home_goals'] > df_train['away_goals']),
        (df_train['home_goals'] == df_train['away_goals']),
        (df_train['home_goals'] < df_train['away_goals'])
    ]
    choices = [2, 1, 0]
    df_train['target'] = np.select(conditions, choices)

    # Encode teams
    le = LabelEncoder()
    all_teams = pd.concat([df_train['home_team'], df_train['away_team']]).unique()
    le.fit(all_teams)
    df_train['home_code'] = le.transform(df_train['home_team'])
    df_train['away_code'] = le.transform(df_train['away_team'])

    # Vectorized Feature Engineering
    # Create a long format DataFrame for calculations
    # Use index to merge back later
    df_train['match_id'] = df_train.index

    # Home perspective
    home_df = df_train[['match_id', 'date', 'home_team', 'home_goals', 'away_goals', 'target']].copy()
    home_df = home_df.rename(columns={'home_team': 'team', 'home_goals': 'goals_for', 'away_goals': 'goals_against'})
    home_df['is_home'] = True
    home_df['points'] = np.where(home_df['target'] == 2, 3, np.where(home_df['target'] == 1, 1, 0))

    # Away perspective
    away_df = df_train[['match_id', 'date', 'away_team', 'away_goals', 'home_goals', 'target']].copy()
    away_df = away_df.rename(columns={'away_team': 'team', 'away_goals': 'goals_for', 'home_goals': 'goals_against'})
    away_df['is_home'] = False
    away_df['points'] = np.where(away_df['target'] == 0, 3, np.where(away_df['target'] == 1, 1, 0))

    # Combine and sort by team and date
    long_df = pd.concat([home_df, away_df]).sort_values(['team', 'date'])

    # Calculate rolling stats
    grouped = long_df.groupby('team')

    # Shift(1) to exclude current match
    # Rolling window of 5 matches
    long_df['avg_goals_for'] = grouped['goals_for'].transform(lambda x: x.shift(1).rolling(window=5, min_periods=1).mean()).fillna(0)
    long_df['form_points'] = grouped['points'].transform(lambda x: x.shift(1).rolling(window=5, min_periods=1).sum()).fillna(0)

    # Pivot back to match_id
    # For home team stats
    home_stats = long_df[long_df['is_home'] == True][['match_id', 'avg_goals_for', 'form_points']]
    home_stats = home_stats.rename(columns={'avg_goals_for': 'home_goals_avg', 'form_points': 'home_form'})

    # For away team stats
    away_stats = long_df[long_df['is_home'] == False][['match_id', 'avg_goals_for', 'form_points']]
    away_stats = away_stats.rename(columns={'avg_goals_for': 'away_goals_avg', 'form_points': 'away_form'})

    # Merge back to df_train
    df_train = df_train.merge(home_stats, on='match_id', how='left')
    df_train = df_train.merge(away_stats, on='match_id', how='left')

    # Fill NaN (first matches)
    df_train[['home_goals_avg', 'home_form', 'away_goals_avg', 'away_form']] = df_train[['home_goals_avg', 'home_form', 'away_goals_avg', 'away_form']].fillna(0)

    feature_cols = ['home_code', 'away_code', 'home_form', 'away_form', 'home_goals_avg', 'away_goals_avg']

    # Create stats dictionary for compatibility with predict.py
    # Structure: {team: {'points': [...], 'goals_for': [...], 'goals_against': [...]}}
    stats = {}
    for team, group in long_df.groupby('team'):
        stats[team] = {
            'points': group['points'].tolist(),
            'goals_for': group['goals_for'].tolist(),
            'goals_against': group['goals_against'].tolist()
        }

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

@lru_cache(maxsize=1)
def load_model(path="./model.pkl"):
    if not os.path.exists(path):
        raise FileNotFoundError(path)
    # Use joblib to load since train_model uses joblib
    try:
        return joblib.load(path)
    except:
        # Fallback to pickle if joblib fails (e.g. for baseline model)
        with open(path, "rb") as f:
            return pickle.load(f)

def predict_score(home, away, model, home_adv=1.1, max_goals=6):
    teams = model["teams"]
    gavg = model["global_avg"]
    home_attack = teams.get(home, {}).get("avg_for", gavg)
    home_def = teams.get(home, {}).get("avg_against", gavg)
    away_attack = teams.get(away, {}).get("avg_for", gavg)
    away_def = teams.get(away, {}).get("avg_against", gavg)
    exp_home = max(0.01, home_attack * (away_def / max(1e-3, gavg)) * home_adv)
    exp_away = max(0.01, away_attack * (home_def / max(1e-3, gavg)))
    # compute most likely score via Poisson probabilities
    best = None
    bestp = 0.0
    for h in range(0, max_goals + 1):
        for a in range(0, max_goals + 1):
            p = poisson.pmf(h, exp_home) * poisson.pmf(a, exp_away)
            if p > bestp:
                bestp = p
                best = (h, a, p)
    return {"home_exp": exp_home, "away_exp": exp_away, "predicted": best}
