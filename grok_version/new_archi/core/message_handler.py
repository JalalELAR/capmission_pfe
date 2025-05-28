# chatbot_project/core/message_handler.py
from llm.llm_agent import call_llm_agent
from core.session_manager import load_context, save_context

def handle_user_message(user_input: str) -> str:
    context = load_context()
    response = call_llm_agent(user_input, context)
    if "context_update" in response:
        save_context(response["context_update"])
    return response.get("message", "Je n'ai pas compris votre demande.")
