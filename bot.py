import time
import yaml
import urllib.request
import urllib.parse
import json
from predict import predict_match


cfg = yaml.safe_load(open("config.yaml"))
TOKEN = cfg["telegram_token"]
BASE = f"https://api.telegram.org/bot{TOKEN}"
OFFSET_FILE = "/tmp/duy_bot_offset.txt"


def get_updates(offset=None, timeout=10):
    url = BASE + f"/getUpdates?timeout={timeout}" + (f"&offset={offset}" if offset else "")
    try:
        with urllib.request.urlopen(url, timeout=timeout + 2) as r:
            return json.load(r)
    except Exception as e:
        return {"ok": False, "error": str(e)}


def send_message(chat_id, text):
    data = urllib.parse.urlencode({"chat_id": chat_id, "text": text}).encode()
    req = urllib.request.Request(BASE + "/sendMessage", data=data, method="POST")
    try:
        with urllib.request.urlopen(req, timeout=10) as r:
            return json.load(r)
    except Exception as e:
        return {"ok": False, "error": str(e)}


def main():
    last = None
    if os.path.exists(OFFSET_FILE := "/tmp/duy_bot_offset.txt"):
        try:
            last = int(open(OFFSET_FILE).read().strip())
        except:
            last = None
    while True:
        updates = get_updates(last + 1 if last else None, timeout=10)
        if not updates.get("ok"):
            time.sleep(2)
            continue
        for u in updates.get("result", []):
            upd = u.get("update_id")
            m = u.get("message") or {}
            chat = m.get("chat", {})
            cid = chat.get("id")
            text = m.get("text", "").strip()
            if not cid or not text:
                last = upd
                continue
            if text.startswith("/predict"):
                parts = text.split()
                if len(parts) >= 3:
                    home = parts[1]
                    away = parts[2]
                    res = predict_match(home, away, model_path=cfg.get("model_path", "./model.pkl"))
                    msg = f'Prediction {home} vs {away}: {res["predicted_score"]} (p={res["probability"]:.3f})\\nexp: {res["home_exp"]:.2f} - {res["away_exp"]:.2f}'
                else:
                    msg = "Usage: /predict HOME_TEAM AWAY_TEAM"
                send_message(cid, msg)
            elif text.startswith("/next"):
                # placeholder: respond with owner note
                send_message(cid, "I will predict next EPL match when pipeline runs. Use /predict HOME AWAY for ad-hoc.")
            else:
                # ignore other messages or confirm
                send_message(cid, "Bot online. Use /predict HOME AWAY to get a score prediction.")
            last = upd
            try:
                open(OFFSET_FILE, "w").write(str(last))
            except:
                pass
        time.sleep(1)


if __name__ == "__main__":
    import os
    main()


