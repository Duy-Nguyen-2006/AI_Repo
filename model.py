import os
import joblib
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
        joblib.dump(model, f)
    return out_path


def prepare_features(schedule_df):
    """
    Vectorized Feature Engineering
    Calculates rolling form and average goals for home and away teams.
    """
    df = schedule_df.copy()

    # Ensure date is datetime
    if 'date' in df.columns:
        df['date'] = pd.to_datetime(df['date'])
        df = df.sort_values('date')

    # Label Encode Teams
    le = LabelEncoder()
    # Fit on all unique teams
    all_teams = pd.concat([df['home_team'], df['away_team']]).unique()
    le.fit(all_teams)

    df['home_code'] = le.transform(df['home_team'])
    df['away_code'] = le.transform(df['away_team'])

    # Calculate Points and Result
    # 0: Away Win, 1: Draw, 2: Home Win
    # Convert goals to numeric first
    df['home_goals'] = pd.to_numeric(df['home_goals'], errors='coerce').fillna(0)
    df['away_goals'] = pd.to_numeric(df['away_goals'], errors='coerce').fillna(0)

    conditions = [
        (df['home_goals'] > df['away_goals']),
        (df['home_goals'] == df['away_goals']),
        (df['home_goals'] < df['away_goals'])
    ]
    values = [2, 1, 0]
    df['target'] = np.select(conditions, values)

    home_points = np.select(conditions, [3, 1, 0])
    away_points = np.select(conditions, [0, 1, 3])

    # Create Long Format for Feature Engineering
    home_df = df[['date', 'home_team', 'home_goals', 'away_goals']].copy()
    home_df.columns = ['date', 'team', 'goals_for', 'goals_against']
    home_df['points'] = home_points
    home_df['is_home'] = True
    home_df['match_index'] = df.index

    away_df = df[['date', 'away_team', 'away_goals', 'home_goals']].copy()
    away_df.columns = ['date', 'team', 'goals_for', 'goals_against']
    away_df['points'] = away_points
    away_df['is_home'] = False
    away_df['match_index'] = df.index

    # Sort by date for rolling calc
    long_df = pd.concat([home_df, away_df]).sort_values(['date'])

    # Group by team and calculate rolling features
    grouped = long_df.groupby('team')

    # Rolling sum of points (Form) - Shift 1 to use past data only
    long_df['form'] = grouped['points'].transform(lambda x: x.shift(1).rolling(5, min_periods=1).sum().fillna(0))

    # Rolling mean of goals_for (Avg Goals) - Shift 1 to use past data only
    long_df['goals_avg'] = grouped['goals_for'].transform(lambda x: x.shift(1).rolling(5, min_periods=1).mean().fillna(0))

    # Restore original index order to map back correctly
    # Note: merge operations might change order, so using match_index is safer

    home_features = long_df[long_df['is_home']].set_index('match_index')[['form', 'goals_avg']]
    home_features.columns = ['home_form', 'home_goals_avg']

    away_features = long_df[~long_df['is_home']].set_index('match_index')[['form', 'goals_avg']]
    away_features.columns = ['away_form', 'away_goals_avg']

    # Join features back to main df
    df = df.join(home_features).join(away_features)

    # Fill remaining NaNs (start of season)
    df = df.fillna(0)

    # Create Stats Dict for Predict (Full history for each team)
    stats = {}
    for team, group in long_df.groupby('team'):
        stats[team] = {
            'points': group['points'].tolist(),
            'goals_for': group['goals_for'].tolist(),
            'goals_against': group['goals_against'].tolist()
        }

    feature_cols = ['home_code', 'away_code', 'home_form', 'away_form', 'home_goals_avg', 'away_goals_avg']

    return df, feature_cols, le, stats


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


def predict_score(home, away, model, home_adv=1.1, max_goals=6):
    # This seems to be for the baseline model, but included for compatibility if needed.
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
