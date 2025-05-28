# llm/llm_agent.py
from openai import OpenAI
from llm.prompt_loader import load_prompt, load_function_definitions

def call_llm_agent(messages: list[dict]) -> dict:
    prompt = load_prompt()
    function_definitions = load_function_definitions()

    response = OpenAI().chat.completions.create(
        model="gpt-4-0613",
        messages=[{"role": "system", "content": prompt}] + messages,
        functions=function_definitions,
        function_call="auto",
        temperature=0.7,
    )
    return response.choices[0].message
