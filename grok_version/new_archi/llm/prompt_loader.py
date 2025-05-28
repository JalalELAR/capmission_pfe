# llm/prompt_loader.py
import json
from pathlib import Path

PROMPT_PATH = Path("prompts/system_prompt.txt")
FUNCTION_DEFS_PATH = Path("prompts/openai_function_definitions.json")

def load_prompt() -> str:
    return PROMPT_PATH.read_text(encoding="utf-8")

def load_function_definitions() -> list:
    if not FUNCTION_DEFS_PATH.exists():
        return []
    return json.loads(FUNCTION_DEFS_PATH.read_text(encoding="utf-8"))
