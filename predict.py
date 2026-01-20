from model import load_model
import numpy as np
import pandas as pd

def predict_match(home, away, model_path="./model.pkl"):
    try:
        data = load_model(model_path)
    except Exception as e:
        return {"error": f"Could not load model: {str(e)}"}

    clf = data["model"]
    le = data["encoder"]
    stats = data["stats"]
    feature_cols = data["feature_cols"]

    # Check if teams exist
    try:
        home_code = le.transform([home])[0]
        away_code = le.transform([away])[0]
    except ValueError:
        return {"error": f"One of the teams ({home}, {away}) is not in the training data."}

    # Calculate features based on latest stats
    # Stats structure: stats[team] = {'points': [...], 'goals_for': [...]}

    def get_avg(team, metric):
        l = stats.get(team, {}).get(metric, [])
        return np.mean(l[-5:]) if len(l) > 0 else 0.0

    def get_form(team):
        l = stats.get(team, {}).get('points', [])
        return sum(l[-5:]) if len(l) > 0 else 0

    home_form = get_form(home)
    away_form = get_form(away)
    home_avg_goals = get_avg(home, 'goals_for')
    away_avg_goals = get_avg(away, 'goals_for')

    # Create DataFrame for prediction
    features = {
        'home_code': [home_code],
        'away_code': [away_code],
        'home_form': [home_form],
        'away_form': [away_form],
        'home_goals_avg': [home_avg_goals],
        'away_goals_avg': [away_avg_goals]
    }

    X = pd.DataFrame(features)

    # Ensure columns order matches training
    X = X[feature_cols]

    # Predict
    probs = clf.predict_proba(X)[0] # [Prob(Away), Prob(Draw), Prob(Home)]
    pred_idx = np.argmax(probs)

    labels = ["Away Win", "Draw", "Home Win"]
    predicted_score = labels[pred_idx] # Note: This is now a result class, not a specific score like 2-1

    # Calculate expected goals (rough estimate based on averages)
    # This is just for display, the model predicts outcome class
    exp_home_goals = home_avg_goals
    exp_away_goals = away_avg_goals

    return {
        "home": home,
        "away": away,
        "home_exp": exp_home_goals,
        "away_exp": exp_away_goals,
        "predicted_score": predicted_score,
        "probability": float(probs[pred_idx]),
        "probs": {
            "home_win": float(probs[2]),
            "draw": float(probs[1]),
            "away_win": float(probs[0])
        }
    }

if __name__ == "__main__":
    import sys
    if len(sys.argv) < 3:
        print("Usage: python predict.py HOME_TEAM AWAY_TEAM")
        sys.exit(1)
    out = predict_match(sys.argv[1], sys.argv[2])
    print(out)
