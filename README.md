# Duy_Football_bot

Lightweight English Premier League score predictor bot.

Quick start
1. Create virtualenv: `python3 -m venv .venv && source .venv/bin/activate`
2. Install deps: `pip install -r requirements.txt`
3. Edit `config.yaml` and set `telegram_token` (bot token) and `owner_chat_id`.
4. Run pipeline once to fetch data and train baseline model:
   `./run_pipeline.sh`
5. Run bot: `python3 bot.py`

Notes
- This project uses `soccerdata` to fetch schedule/results and a lightweight Poisson-based model as a baseline.
- Designed to be small and runnable on a modest machine; heavy training is not included.


