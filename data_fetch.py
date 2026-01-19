import os
import json
from datetime import datetime
import pandas as pd

try:
    import soccerdata as sd
except Exception:
    sd = None


def ensure_dir(p):
    os.makedirs(p, exist_ok=True)


def fetch_schedule(season=None, league="ENG-Premier League", out_dir="./data"):
    ensure_dir(out_dir)
    if sd is None:
        raise RuntimeError("soccerdata not installed")
    fb = sd.FBref(league, season) if season else sd.FBref(league)
    sched = fb.read_schedule()
    path = os.path.join(out_dir, "schedule.parquet")
    sched.to_parquet(path)
    return path


def fetch_team_stats(season=None, league="ENG-Premier League", out_dir="./data"):
    ensure_dir(out_dir)
    if sd is None:
        raise RuntimeError("soccerdata not installed")
    fb = sd.FBref(league, season) if season else sd.FBref(league)
    team_stats = fb.read_team_season_stats(stat_type="standard")
    path = os.path.join(out_dir, "team_stats.parquet")
    team_stats.to_parquet(path)
    return path


def load_schedule(path="./data/schedule.parquet"):
    if not os.path.exists(path):
        raise FileNotFoundError(path)
    return pd.read_parquet(path)


def load_team_stats(path="./data/team_stats.parquet"):
    if not os.path.exists(path):
        raise FileNotFoundError(path)
    return pd.read_parquet(path)


