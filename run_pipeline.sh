#!/usr/bin/env bash
set -e
python3 - <<'PY'
from data_fetch import fetch_schedule, fetch_team_stats
from model import train_baseline
import yaml, os
cfg=yaml.safe_load(open('config.yaml'))
outdir=cfg.get('data_cache','./data')
print('fetching schedule...')
try:
    fetch_schedule(season=None, league=cfg.get('league','ENG-Premier League'), out_dir=outdir)
except Exception as e:
    print('fetch schedule failed:', e)
print('fetching team stats...')
try:
    fetch_team_stats(season=None, league=cfg.get('league','ENG-Premier League'), out_dir=outdir)
except Exception as e:
    print('fetch team stats failed:', e)
import pandas as pd
try:
    sched=pd.read_parquet(os.path.join(outdir,'schedule.parquet'))\nexcept Exception as e:\n    print('no schedule parquet', e); sched=pd.DataFrame()\nif not sched.empty:\n    print('training baseline model...')\n    train_baseline(sched, out_path=cfg.get('model_path','./model.pkl'))\nelse:\n    print('no schedule to train on')\nPY
echo "Pipeline finished."

