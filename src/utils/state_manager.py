import json
import os
import logging
from pathlib import Path

logger = logging.getLogger("STATE_MANAGER")

STATE_FILE = "data/system_state.json"

def save_system_state(data: dict):
    """
    Persist critical system metrics to JSON using atomic write.

    Writes to a .tmp file first, then os.replace() swaps it in one
    filesystem operation. The live file is never partially written.
    If the process is killed mid-write, the .tmp is orphaned but the
    previous good state file is untouched.
    """
    try:
        state_path = Path(STATE_FILE)
        state_path.parent.mkdir(exist_ok=True)
        tmp_path = state_path.with_suffix(".tmp")
        with open(tmp_path, "w") as f:
            json.dump(data, f, indent=4)
        os.replace(tmp_path, state_path)  # Atomic on POSIX; near-atomic on Windows
        logger.info(f"[STATE] System state saved to {STATE_FILE}")
    except Exception as e:
        logger.error(f"[STATE] Failed to save state: {e}")

def load_system_state() -> dict:
    """
    Restore critical system metrics from JSON.
    """
    if not os.path.exists(STATE_FILE):
        logger.info("[STATE] No system state file found.")
        return {}
        
    try:
        with open(STATE_FILE, "r") as f:
            state = json.load(f)
        logger.info(f"[STATE] System state restored from {STATE_FILE}")
        return state
    except Exception as e:
        logger.error(f"[STATE] Failed to load state: {e}")
        return {}
