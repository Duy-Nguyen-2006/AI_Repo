import os
import joblib
import numpy as np
import pandas as pd
from scipy.stats import poisson
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import LabelEncoder

def train_baseline(schedule_df, out_path="./model.pkl"):
    # Simple home/away attack-defense rates from historical matches in schedule_df
    # Expects schedule_df with columns: home_team, away_team, home_goals, away_goals

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
    total_stats = home_stats.add(away_stats, fill_value=0)
    total_stats = total_stats.rename(columns={"goals_for": "for", "goals_against": "against"})

    df = total_stats

    # Avoid division by zero
    df["avg_for"] = df["for"] / df["played"].replace(0, 1)
    df["avg_against"] = df["against"] / df["played"].replace(0, 1)

    # global average
    gavg = df["avg_for"].mean()

    # Convert to dictionary with "teams" key
    model = {"teams": df.to_dict(orient="index"), "global_avg": float(gavg)}

    joblib.dump(model, out_path)
    return out_path

def prepare_features(schedule_df):
    """
    Vectorized feature engineering.
    Calculates rolling form (sum of points) and goal averages (mean of goals scored)
    for the last 5 matches, EXCLUDING the current match.
    """
    df = schedule_df.copy()

    # Ensure date is datetime
    df['date'] = pd.to_datetime(df['date'])
    # Sort by date
    df = df.sort_values('date')

    le = LabelEncoder()
    # Fit on all unique teams
    all_teams = pd.concat([df['home_team'], df['away_team']]).unique()
    le.fit(all_teams)

    df['home_code'] = le.transform(df['home_team'])
    df['away_code'] = le.transform(df['away_team'])

    # Calculate points for each match to build history
    # Target: 0=Away Win, 1=Draw, 2=Home Win
    conditions = [
        (df['home_goals'] > df['away_goals']),
        (df['home_goals'] == df['away_goals']),
        (df['home_goals'] < df['away_goals'])
    ]
    df['target'] = np.select(conditions, [2, 1, 0])

    # Points: Win=3, Draw=1, Loss=0
    df['home_points'] = np.select(conditions, [3, 1, 0])
    df['away_points'] = np.select(conditions, [0, 1, 3])

    # Create Long Format DataFrame for vectorized rolling calculations
    home_df = df[['date', 'home_team', 'home_points', 'home_goals', 'away_goals']].rename(columns={
        'home_team': 'team', 'home_points': 'points', 'home_goals': 'goals_for', 'away_goals': 'goals_against'
    })
    away_df = df[['date', 'away_team', 'away_points', 'away_goals', 'home_goals']].rename(columns={
        'away_team': 'team', 'away_points': 'points', 'away_goals': 'goals_for', 'home_goals': 'goals_against'
    })

    long_df = pd.concat([home_df, away_df]).sort_values(['date', 'team']).reset_index(drop=True)

    # Calculate rolling stats
    # closed='left' ensures we use previous matches only
    # min_periods=1 allows stats even after 1 match
    rolled = long_df.groupby('team')[['points', 'goals_for']].rolling(window=5, min_periods=1, closed='left').agg({
        'points': 'sum',
        'goals_for': 'mean'
    })

    # The result 'rolled' has MultiIndex (team, original_index)
    # We map it back to long_df using the original_index (level 1)
    long_df['form'] = rolled['points'].reset_index(level=0, drop=True)
    long_df['goals_avg'] = rolled['goals_for'].reset_index(level=0, drop=True)

    # Fill NaN (start of season/no history) with 0
    long_df['form'] = long_df['form'].fillna(0)
    long_df['goals_avg'] = long_df['goals_avg'].fillna(0)

    # Now merge these features back to the main DataFrame

    # Merge for Home Team
    df = df.merge(
        long_df[['date', 'team', 'form', 'goals_avg']],
        left_on=['date', 'home_team'],
        right_on=['date', 'team'],
        how='left'
    ).rename(columns={'form': 'home_form', 'goals_avg': 'home_goals_avg'}).drop(columns=['team'])

    # Merge for Away Team
    df = df.merge(
        long_df[['date', 'team', 'form', 'goals_avg']],
        left_on=['date', 'away_team'],
        right_on=['date', 'team'],
        how='left'
    ).rename(columns={'form': 'away_form', 'goals_avg': 'away_goals_avg'}).drop(columns=['team'])

    feature_cols = ['home_code', 'away_code', 'home_form', 'away_form', 'home_goals_avg', 'away_goals_avg']

    # Generate stats dictionary for predict.py (needs full history)
    stats = {}
    for team, group in long_df.groupby('team'):
        stats[team] = {
            'points': group['points'].tolist(),
            'goals_for': group['goals_for'].tolist(),
            'goals_against': group['goals_against'].tolist()
        }

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
