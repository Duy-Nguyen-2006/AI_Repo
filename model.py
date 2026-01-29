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
    # Ensure date is datetime
    schedule_df = schedule_df.copy()
    schedule_df['date'] = pd.to_datetime(schedule_df['date'])
    schedule_df = schedule_df.sort_values('date')

    # Label Encode Teams
    le = LabelEncoder()
    # Fit on all unique teams in home and away columns
    all_teams = pd.unique(pd.concat([schedule_df['home_team'], schedule_df['away_team']]))
    le.fit(all_teams)

    schedule_df['home_code'] = le.transform(schedule_df['home_team'])
    schedule_df['away_code'] = le.transform(schedule_df['away_team'])

    # Determine match results and points
    # Target: 0=Away Win, 1=Draw, 2=Home Win
    conditions = [
        schedule_df['home_goals'] > schedule_df['away_goals'],
        schedule_df['home_goals'] == schedule_df['away_goals'],
        schedule_df['home_goals'] < schedule_df['away_goals']
    ]
    choices = [2, 1, 0]
    schedule_df['target'] = np.select(conditions, choices)

    # Points for home and away
    # Home: Win=3, Draw=1, Loss=0
    home_points = np.select(conditions, [3, 1, 0])
    # Away: Win=0, Draw=1, Loss=3 (relative to home result)
    # Wait, if Home Win (idx 0), Away gets 0. If Draw (idx 1), Away gets 1. If Home Loss (idx 2), Away Win, so 3.
    away_points = np.select(conditions, [0, 1, 3])

    # Create Long Format for Rolling Stats
    # We want two rows per match: one for home team perspective, one for away team perspective

    home_df = schedule_df[['date', 'home_team', 'home_goals', 'away_goals']].copy()
    home_df.columns = ['date', 'team', 'goals_for', 'goals_against']
    home_df['points'] = home_points

    away_df = schedule_df[['date', 'away_team', 'away_goals', 'home_goals']].copy()
    away_df.columns = ['date', 'team', 'goals_for', 'goals_against']
    away_df['points'] = away_points

    # Combine and sort by team then date to ensure rolling works correctly
    long_df = pd.concat([home_df, away_df], ignore_index=True)
    long_df = long_df.sort_values(['team', 'date'])

    # Calculate Rolling Features (shifted by 1 to prevent leakage)
    grouped = long_df.groupby('team')

    # Form: Sum of points in last 5 games
    # shift(1) means the value at index i is the sum of i-5...i-1.
    # i.e. stats BEFORE the current match.
    long_df['form'] = grouped['points'].transform(
        lambda x: x.rolling(window=5, min_periods=1).sum().shift(1).fillna(0)
    )

    # Avg Goals: Mean of goals_for in last 5 games
    long_df['goals_avg'] = grouped['goals_for'].transform(
        lambda x: x.rolling(window=5, min_periods=1).mean().shift(1).fillna(0)
    )

    # Now merge these stats back to the original schedule_df.
    # We need to map back based on team and date.

    # Since we have duplicate dates potentially (though rare for same team), but 'team' + 'date' should be unique per match per team.

    # Create temp dfs for merging
    # Home stats
    feat_home = long_df[['date', 'team', 'form', 'goals_avg']].rename(columns={
        'team': 'home_team',
        'form': 'home_form',
        'goals_avg': 'home_goals_avg'
    })

    # Away stats
    feat_away = long_df[['date', 'team', 'form', 'goals_avg']].rename(columns={
        'team': 'away_team',
        'form': 'away_form',
        'goals_avg': 'away_goals_avg'
    })

    # Merge
    # We use left join to keep all matches in schedule_df
    df_train = pd.merge(schedule_df, feat_home, on=['date', 'home_team'], how='left')
    df_train = pd.merge(df_train, feat_away, on=['date', 'away_team'], how='left')

    feature_cols = ['home_code', 'away_code', 'home_form', 'away_form', 'home_goals_avg', 'away_goals_avg']

    # Reconstruct stats dictionary for predict.py
    # predict.py needs the full history to calculate stats for the NEXT match.
    # So we give it the lists from long_df.
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


def load_model(path="./model.pkl"):
    if not os.path.exists(path):
        raise FileNotFoundError(path)
    with open(path, "rb") as f:
        # Try joblib first since we use it in train_model
        try:
            return joblib.load(f)
        except:
            # Fallback to pickle for backward compatibility or if joblib fails
            f.seek(0)
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
