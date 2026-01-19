import os
import pickle
import numpy as np
import pandas as pd
from scipy.stats import poisson


def train_baseline(schedule_df, out_path="./model.pkl"):
    # Simple home/away attack-defense rates from historical matches in schedule_df
    # Expects schedule_df with columns: home_team, away_team, home_goals, away_goals
    teams = pd.concat([schedule_df["home_team"], schedule_df["away_team"]]).unique()
    stats = {t: {"for": 0.0, "against": 0.0, "played": 0} for t in teams}
    for _, r in schedule_df.iterrows():
        ht, at = r["home_team"], r["away_team"]
        hg, ag = int(r.get("home_goals", 0)), int(r.get("away_goals", 0))
        stats[ht]["for"] += hg
        stats[ht]["against"] += ag
        stats[ht]["played"] += 1
        stats[at]["for"] += ag
        stats[at]["against"] += hg
        stats[at]["played"] += 1
    df = pd.DataFrame.from_dict(stats, orient="index")
    # Avoid division by zero
    df["avg_for"] = df["for"] / df["played"].replace(0, 1)
    df["avg_against"] = df["against"] / df["played"].replace(0, 1)
    # global average
    gavg = df["avg_for"].mean()
    model = {"teams": df.to_dict(orient="index"), "global_avg": float(gavg)}
    with open(out_path, "wb") as f:
        pickle.dump(model, f)
    return out_path


def load_model(path="./model.pkl"):
    if not os.path.exists(path):
        raise FileNotFoundError(path)
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


