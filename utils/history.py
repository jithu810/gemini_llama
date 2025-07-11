import os
from utils.config import Config

logger = Config.init_logging()
loggers = logger['history']

def load_history(query_id):
        path = f"./history/{query_id}.json"
        if os.path.exists(path):
            try:
                import json
                with open(path, "r") as f:
                    data = json.load(f)
                    loggers.info(f"[HISTORY] Loaded {len(data)} messages from history.")
                    return data, None
            except Exception as e:
                loggers.error(f"[HISTORY LOAD ERROR] {e}")
                return [], f"Failed to load history: {str(e)}"
        else:
            loggers.info("[HISTORY] No existing history found.")
            return [], None
        
def save_history(messages,query_id):
    path = f"./history/{query_id}.json"
    os.makedirs(os.path.dirname(path), exist_ok=True)
    try:
        import json
        with open(path, "w") as f:
            json.dump(messages, f, indent=2)
        loggers.info(f"[HISTORY] Saved {len(messages)} messages to history.")
        return True, None
    except Exception as e:
        loggers.error(f"[HISTORY SAVE ERROR] {e}")
        return False, str(e)