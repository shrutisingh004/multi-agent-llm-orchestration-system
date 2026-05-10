from google import genai
from app.config import GEMINI_API_KEY, GEMINI_MODEL

_client = None

def get_client():
    global _client
    if _client is None:
        _client = genai.Client(api_key=GEMINI_API_KEY)
    return _client

async def call_llm(prompt: str, system: str = "") -> str:
    contents = f"{system}\n\n{prompt}" if system else prompt
    response = await get_client().aio.models.generate_content(
        model=GEMINI_MODEL,
        contents=contents,
    )
    return response.text or ""

async def call_llm_json(prompt: str, system: str = "") -> str:
    sys_str = (system + "\n\nRespond ONLY with valid JSON. No markdown, no backticks.") if system else "Respond ONLY with valid JSON. No markdown, no backticks."
    return await call_llm(prompt, sys_str)
