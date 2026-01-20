#!/usr/bin/env bash
set -e
python3 - <<'PY'
from data_fetch import fetch_schedule
from model import train_model
import yaml, os
import pandas as pd

cfg=yaml.safe_load(open('config.yaml'))
outdir=cfg.get('data_cache','./data')

print('fetching schedule for multiple seasons...')
try:
    # Fetch historical data (last 5 years + current)
    # The default behavior of updated fetch_schedule handles the list of seasons
    fetch_schedule(season=None, league=cfg.get('league','ENG-Premier League'), out_dir=outdir)
except Exception as e:
    print('fetch schedule failed:', e)

try:
    sched=pd.read_parquet(os.path.join(outdir,'schedule.parquet'))
except Exception as e:
    print('no schedule parquet', e); sched=pd.DataFrame()

if not sched.empty:
    print('training ML model...')
    train_model(sched, out_path=cfg.get('model_path','./model.pkl'))
else:
    print('no schedule to train on')
PY
echo "Pipeline finished."
