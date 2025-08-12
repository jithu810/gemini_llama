import os
import json

class ChatHistoryManager:
    def __init__(self, history_dir="./history"):
        self.history_dir = history_dir
        os.makedirs(self.history_dir, exist_ok=True)

    def load_history(self, query_id):
        """Load chat history from JSON file."""
        path = os.path.join(self.history_dir, f"{query_id}.json")
        
        if os.path.exists(path):
            try:
                with open(path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                return data, None
            except Exception as e:
                return [], f"Failed to load history: {str(e)}"
        else:
            return [], None

    def save_history(self, query_id, messages):
        """Save chat history to JSON file."""
        path = os.path.join(self.history_dir, f"{query_id}.json")
        
        try:
            with open(path, "w", encoding="utf-8") as f:
                json.dump(messages, f, indent=2, ensure_ascii=False)
            return True, None
        except Exception as e:
            return False, str(e)
