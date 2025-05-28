# chatbot_project/orchestrator.py

from core.session_manager import load_user_state, save_user_state
from llm.llm_agent import call_llm_agent
from tools.functions import *
from tools.chroma_utils import *
import json

def orchestrate_message_flow(user_input: str, messages: list) -> tuple[str, dict]:
    """
    Orchestration logic that handles user input, calls the LLM with context, manages state and functions.
    """
    # Load current user state (shared JSON)
    user_state = load_user_state()

    # Add user's message to history
    messages.append({"role": "user", "content": user_input})

    # Call LLM (with function calling capabilities)
    llm_response = call_llm_agent(messages)

    # If LLM wants to call a function
    if hasattr(llm_response, "function_call"):
        func_name = llm_response.function_call.name
        func_args = json.loads(llm_response.function_call.arguments)

        # Map function names to actual implementations
        function_map = {
            "get_available_forfaits": get_available_forfaits,
            "get_types_duree_forfait": get_types_duree_forfait,
            "get_remaining_sessions": get_remaining_sessions,
            "calculate_tariffs": calculate_tariffs,
            "check_overlaps": check_overlaps,
            "match_value": match_value,
            "get_recommendations": get_recommendations
        }

        if func_name in function_map:
            try:
                # Call the function and return the result to the LLM
                result = function_map[func_name](**func_args)
                messages.append({
                    "role": "function",
                    "name": func_name,
                    "content": json.dumps(result, ensure_ascii=False)
                })

                # Recursive call to LLM with function result included
                follow_up = call_llm_agent(messages)
                messages.append({"role": "assistant", "content": follow_up.content})

                return follow_up.content, user_state

            except Exception as e:
                error_msg = f"[Erreur fonction {func_name}] : {str(e)}"
                messages.append({"role": "assistant", "content": error_msg})
                return error_msg, user_state
        else:
            fallback_msg = f"La fonction {func_name} n'est pas reconnue."
            messages.append({"role": "assistant", "content": fallback_msg})
            return fallback_msg, user_state

    # If LLM just replies normally
    if hasattr(llm_response, "content"):
        messages.append({"role": "assistant", "content": llm_response.content})
        return llm_response.content, user_state

    return "[Réponse vide du LLM]", user_state