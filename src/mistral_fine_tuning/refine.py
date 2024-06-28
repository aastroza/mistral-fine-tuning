from mistralai.async_client import MistralAsyncClient
import instructor
from pydantic import BaseModel, Field
from typing import List
from enum import Enum

from dotenv import load_dotenv
load_dotenv()

mistral_client = MistralAsyncClient()
patched_client = instructor.from_mistral(client=mistral_client, mode=instructor.Mode.MISTRAL_TOOLS)

class Quality(Enum):
    FUNNY = "funny"
    NOT_FUNNY = "not_funny"

class Refinement(BaseModel):
    text: str = Field(..., description="The corrected joke text. Fix grammar. Fix typos and missing characters. Ensure correct use of punctuation.")    
    keywords: List[str] = Field(..., default_factory=list, description="The corrected keywords array")
    quality: Quality = Field(..., description="The quality of the joke")

SYSTEM_PROMPT = """
You are tasked with correcting jokes to ensure they are well-written and funny.
Correct the joke text by fixing grammar. Fix typos and missing characters. Ensure correct use of punctuation.
Also correct the keywords array fixing any mistakes or typos.
Review the corrected joke text and determine if it is funny or not.
"""

async def refine_joke(text: str, keywords: str, language: str) -> Refinement:
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
             "content": f"Help me refine the following joke: '{text}'\n and keywords: {keywords}",
             },
        ],
        temperature=0,
        response_model=Refinement,
    )