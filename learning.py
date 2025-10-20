# learning.py
import json
import os

STATS_FILE = "learning_stats.json"

def load_stats():
    if os.path.exists(STATS_FILE):
        try:
            with open(STATS_FILE, "r") as f:
                return json.load(f)
        except:
            return {"success": 0, "fail": 0}
    return {"success": 0, "fail": 0}

def save_stats(stats):
    with open(STATS_FILE, "w") as f:
        json.dump(stats, f)

def record_result(success: bool):
    stats = load_stats()
    if success:
        stats["success"] += 1
    else:
        stats["fail"] += 1
    save_stats(stats)

def get_confidence_factor() -> float:
    stats = load_stats()
    total = stats["success"] + stats["fail"]
    if total == 0:
        return 1.0
    ratio = stats["success"] / total
    return 0.8 + 0.4 * ratio  # 0.8–1.2 коефіцієнт адаптації
