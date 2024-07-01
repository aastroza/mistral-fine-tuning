from mistralai.async_client import MistralAsyncClient
from openai import AsyncOpenAI
import instructor
from pydantic import BaseModel, Field
from typing import List
from enum import Enum
from loguru import logger

from dotenv import load_dotenv
load_dotenv()

client = instructor.patch(AsyncOpenAI())

mistral_client = MistralAsyncClient()
patched_client = instructor.from_mistral(client=mistral_client, mode=instructor.Mode.MISTRAL_TOOLS)

class Quality(Enum):
    FUNNY = "funny"
    NOT_FUNNY = "not_funny"

class Refinement(BaseModel):
    text: str = Field(..., description="The corrected joke text. Fix grammar. Fix typos and missing characters. Ensure correct use of punctuation.")    
    keywords: List[str] = Field(..., default_factory=list, description="The corrected keywords array")
    quality: Quality = Field(..., description="The quality of the joke")

class Rewrite(BaseModel):
    rewritten_text: str = Field(..., description="The rewritten joke. Maintaining the original humorous style but improving its grammar, punctuation, and presentation.")

SYSTEM_PROMPT = """
You are tasked with correcting jokes to ensure they are well-written and funny.
Correct the joke text by fixing grammar. Fix typos and missing characters. Ensure correct use of punctuation.
Also correct the keywords array fixing any mistakes or typos.
Review the corrected joke text and determine if it is funny or not.
"""

SYSTEM_PROMPT_REWRITING = """
You are a world-class editor in Chilean Spanish.
Correct this joke while maintaining the original humorous style but improving its grammar, punctuation, and presentation.
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

async def rewrite_joke(text: str, language: str) -> Rewrite:
    try:
        return await client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {
                    "role": "system",
                    "content": SYSTEM_PROMPT_REWRITING
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
                "content": f"{text}",
                },
            ],
            temperature=1.2,
            response_model=Rewrite,
        )
    except Exception as e:
        logger.error(f"Error rewriting joke: {e}, trying to rewrite joke with GPT-4-turbo.")
        try:
            return await client.chat.completions.create(
                model="gpt-4-turbo",
                messages=[
                    {
                        "role": "system",
                        "content": SYSTEM_PROMPT_REWRITING
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
                    "content": f"{text}",
                    },
                ],
                temperature=0.8,
                response_model=Rewrite,
            )
        except Exception as e:
            logger.error(f"Error rewriting joke: {e}.")
            return Rewrite(rewritten_text="")
