import os
import joblib
import pickle
import numpy as np
import pandas as pd
from scipy.stats import poisson
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

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

    # Ensure goals are numeric
    df_train["home_goals"] = pd.to_numeric(df_train["home_goals"], errors="coerce").fillna(0).astype(int)
    df_train["away_goals"] = pd.to_numeric(df_train["away_goals"], errors="coerce").fillna(0).astype(int)

    # Encode teams
    le = LabelEncoder()
    # Fit on all unique teams
    all_teams = pd.concat([df_train['home_team'], df_train['away_team']]).unique()
    le.fit(all_teams)

    df_train['home_code'] = le.transform(df_train['home_team'])
    df_train['away_code'] = le.transform(df_train['away_team'])

    # Create target
    # 0: Home Win, 1: Draw, 2: Away Win
    conditions = [
        (df_train['home_goals'] > df_train['away_goals']),
        (df_train['home_goals'] == df_train['away_goals']),
        (df_train['home_goals'] < df_train['away_goals'])
    ]
    choices = [0, 1, 2]
    df_train['target'] = np.select(conditions, choices)

    # Initialize stats for feature calculation
    stats = {team: {'points': [], 'goals_for': [], 'goals_against': []} for team in all_teams}

    # Vectorized feature calculation
    # Create long-form DF (team, points, goals_for) sorted by index
    home_df = df_train[['home_team', 'home_goals', 'away_goals']].copy()
    home_df.columns = ['team', 'goals_for', 'goals_against']
    home_df['is_home'] = True
    home_df['match_index'] = df_train.index

    away_df = df_train[['away_team', 'away_goals', 'home_goals']].copy()
    away_df.columns = ['team', 'goals_for', 'goals_against']
    away_df['is_home'] = False
    away_df['match_index'] = df_train.index

    team_matches = pd.concat([home_df, away_df]).sort_values(by='match_index')

    # Calculate points
    conditions_pts = [
        (team_matches['goals_for'] > team_matches['goals_against']),
        (team_matches['goals_for'] == team_matches['goals_against'])
    ]
    choices_pts = [3, 1]
    team_matches['points'] = np.select(conditions_pts, choices_pts, default=0)

    # Group by team
    grouped = team_matches.groupby('team')

    # Calculate rolling stats (shifted by 1 to exclude current match)
    # Form: Sum of points in last 5 matches
    team_matches['form'] = grouped['points'].transform(
        lambda x: x.shift(1).rolling(window=5, min_periods=0).sum().fillna(0)
    )

    # Avg Goals: Mean of goals_for in last 5 matches (matching predict.py logic)
    team_matches['avg_goals'] = grouped['goals_for'].transform(
        lambda x: x.shift(1).rolling(window=5, min_periods=0).mean().fillna(0)
    )

    # Re-populate stats dict for return (needed for inference)
    for team, group in grouped:
        stats[team]['points'] = group['points'].tolist()
        stats[team]['goals_for'] = group['goals_for'].tolist()
        stats[team]['goals_against'] = group['goals_against'].tolist()

    # Merge features back to df_train
    # Home features
    home_feats = team_matches[team_matches['is_home']].set_index('match_index')[['form', 'avg_goals']]
    df_train['home_form'] = home_feats['form']
    df_train['home_goals_avg'] = home_feats['avg_goals']

    # Away features
    away_feats = team_matches[~team_matches['is_home']].set_index('match_index')[['form', 'avg_goals']]
    df_train['away_form'] = away_feats['form']
    df_train['away_goals_avg'] = away_feats['avg_goals']

    # Fill NaNs
    df_train.fillna(0, inplace=True)

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
    # Try joblib first as it is preferred for sklearn models
    try:
        return joblib.load(path)
    except:
        with open(path, "rb") as f:
            return pickle.load(f)


def predict_score(home, away, model, home_adv=1.1, max_goals=6):
    teams = model.get("teams", {})
    gavg = model.get("global_avg", 1.0)
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
