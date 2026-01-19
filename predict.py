from model import load_model, predict_score


def predict_match(home, away, model_path="./model.pkl"):
    model = load_model(model_path)
    res = predict_score(home, away, model)
    h, a, p = res["predicted"]
    return {
        "home": home,
        "away": away,
        "home_exp": res["home_exp"],
        "away_exp": res["away_exp"],
        "predicted_score": f"{h}-{a}",
        "probability": p,
    }

if __name__ == "__main__":
    import sys
    if len(sys.argv) < 3:
        print("Usage: python predict.py HOME_TEAM AWAY_TEAM")
        sys.exit(1)
    out = predict_match(sys.argv[1], sys.argv[2])
    print(out)


