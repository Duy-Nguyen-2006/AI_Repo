import os
import json
from datetime import datetime
import pandas as pd
import requests
import io

try:
    import soccerdata as sd
except Exception:
    sd = None


def ensure_dir(p):
    os.makedirs(p, exist_ok=True)

def fetch_from_football_data_co_uk(seasons, out_dir):
    """
    Fallback: Download CSVs from football-data.co.uk
    URL format: https://www.football-data.co.uk/mmz4281/2324/E0.csv
    """
    print("Attempting to fetch from football-data.co.uk...")
    dfs = []
    for season in seasons:
        # season is "1819", "1920" etc.
        url = f"https://www.football-data.co.uk/mmz4281/{season}/E0.csv"
        try:
            print(f"Downloading {url}...")
            s = requests.get(url).content
            df = pd.read_csv(io.StringIO(s.decode('utf-8', errors='ignore')))
            # Standardize columns
            # football-data.co.uk: Date, HomeTeam, AwayTeam, FTHG, FTAG
            df = df[['Date', 'HomeTeam', 'AwayTeam', 'FTHG', 'FTAG']].copy()
            df.rename(columns={
                'HomeTeam': 'home_team',
                'AwayTeam': 'away_team',
                'FTHG': 'home_goals',
                'FTAG': 'away_goals',
                'Date': 'date'
            }, inplace=True)
            # Date format usually dd/mm/yy or dd/mm/yyyy
            df['date'] = pd.to_datetime(df['date'], dayfirst=True)
            dfs.append(df)
        except Exception as e:
            print(f"Failed to fetch {season}: {e}")

    if not dfs:
        raise RuntimeError("No data could be fetched from football-data.co.uk")

    full_df = pd.concat(dfs, ignore_index=True)
    path = os.path.join(out_dir, "schedule.parquet")
    full_df.to_parquet(path)
    return path

def fetch_schedule(season=None, league="ENG-Premier League", out_dir="./data"):
    ensure_dir(out_dir)

    # If season is not provided, fetch last 5 seasons
    if season is None:
        # For football-data.co.uk, format is 2324 for 2023/2024
        # soccerdata might use "2324" or "2023-2024"
        seasons = ["1819", "1920", "2021", "2122", "2223", "2324", "2425"]
    elif isinstance(season, list):
        seasons = season
    else:
        seasons = [season]

    print(f"Fetching data for seasons: {seasons}")

    # Try soccerdata first (if installed and working)
    try:
        if sd is None:
            raise RuntimeError("soccerdata not installed")
        fb = sd.FBref(league, seasons)
        sched = fb.read_schedule()
        path = os.path.join(out_dir, "schedule.parquet")
        sched.to_parquet(path)
        return path
    except Exception as e:
        print(f"soccerdata fetch failed ({e}). Switching to fallback...")
        return fetch_from_football_data_co_uk(seasons, out_dir)


def fetch_team_stats(season=None, league="ENG-Premier League", out_dir="./data"):
    # This is less critical for the current model
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
