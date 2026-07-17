from dotenv import load_dotenv
from google import genai
import os

DEFAULT_MODEL = "gemini-3.1-flash-lite"


def main(sys_prompt):
    load_dotenv()

    gemini_api_key = os.getenv("GEMINI_API_KEY")
    if not gemini_api_key:
        return "GEMINI_API_KEY is not configured."

    client = genai.Client(api_key=gemini_api_key)
    model = os.getenv("GEMINI_MODEL", DEFAULT_MODEL)

    try:
        response = client.models.generate_content(
            model=model,
            contents=sys_prompt
        )
    except Exception as error:
        return f"Gemini generation failed: {error}"

    return getattr(response, "text", "") or "No response received from Gemini."
