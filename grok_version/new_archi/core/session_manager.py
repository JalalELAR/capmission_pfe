# chatbot_project/core/session_manager.py
import json
from pathlib import Path
from schemas.user_state_model import UserState

CTX_FILE = Path("data/user_state.json")

def load_context() -> dict:
    if not CTX_FILE.exists():
        return UserState().dict()
    data = json.loads(CTX_FILE.read_text(encoding="utf-8"))
    return UserState(**data).dict()

def save_context(context_update: dict):
    current_state = load_context()
    current_state.update(context_update)
    valid_state = UserState(**current_state)
    CTX_FILE.write_text(valid_state.model_dump_json(indent=2, ensure_ascii=False), encoding="utf-8")

# core/session_manager.py

import json
from pathlib import Path
from schemas.user_state_model import UserState

CTX_FILE = Path("data/user_state.json")

def load_user_state() -> UserState:
    """Charge l'état utilisateur depuis le fichier JSON"""
    if not CTX_FILE.exists():
        return UserState()
    data = json.loads(CTX_FILE.read_text(encoding="utf-8"))
    return UserState(**data)

def save_user_state(state: UserState):
    """Sauvegarde l'état utilisateur dans le fichier JSON"""
    with open(CTX_FILE, "w", encoding="utf-8") as f:
        json.dump(state.dict(), f, indent=2, ensure_ascii=False)
