from mistralai.async_client import MistralAsyncClient
import instructor
from pydantic import BaseModel, Field
from typing import List

from dotenv import load_dotenv
load_dotenv()

mistral_client = MistralAsyncClient()
patched_client = instructor.from_mistral(client=mistral_client, mode=instructor.Mode.MISTRAL_TOOLS)

class Topics(BaseModel):
    keywords: List[str] = Field(..., default_factory=list, description="The extracted keywords from the joke transcript. Extract at least 1 keyword and no more than 5.")

SYSTEM_PROMPT = """
You are tasked with analyzing jokes to identify the topics or themes involved using keyword extraction.
"""

async def extract_keywords(text: str, language: str) -> Topics:
    return await patched_client.chat.completions.create(
        model="mistral-large-latest",
        messages=[
            {
                "role": "system",
                "content": SYSTEM_PROMPT
            },
            {
                "role": "user",
                "content": f"I have added a feature that forces you to response only in `locale={language}` and consider only chilean spanish.",
            },
            {
                "role": "assistant",
                "content": f"Understood thank you. From now I will only response with `locale={language}`",
            },
            {"role": "user",
             "content": f"Help me understand the following joke by extracting keywords: {text}"
             },
        ],
        temperature=0.8,
        response_model=Topics,
    )