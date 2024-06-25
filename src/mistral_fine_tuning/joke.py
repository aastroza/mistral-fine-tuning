import instructor
from pydantic import BaseModel, Field
from typing import List
from loguru import logger
from mistralai.client import MistralClient

from dotenv import load_dotenv
load_dotenv()

SYSTEM_PROMPT = """
You are professional comedian writer tasked with extracting a clean list of jokes from a given comedy routine transcript.
The jokes must be structured in a clear and precise manner that makes use of timestamps, when available, to help others study the routine.
"""

USER_PROMPT = """
Extract a list of jokes from the transcript in a clear and concise manner to help others study the jokes from the comedy routine. Respond in the same language as the transcript if it is not english.

Extraction Tips:
* Do not extract content if it only involves music or if nothing happens; do not include these in the jokes.
* Use only content from the transcript. Do not add any additional information.
* The original comedy routine transcript was automatically generated and may be messy. Clean the jokes in the transcript.
* Only include completed jokes.
"""


class Joke(BaseModel):
    transcript: str = Field(
        description="The joke transcript. Do not include comments, greetings, or any other non-joke content."
    )
    corrected_transcript: str = Field(
        description="The corrected joke transcript. Clean the transcript from any unnecessary content. Fix typos. Ensure correct use of punctuation. Make sure the joke is clean like a historical quote."
    )

class Repertoire(BaseModel):
    jokes: List[Joke]


def create_jokes_from_transcript(txt: str, language: str = "es") -> Repertoire:
    
    client = MistralClient()

    patched_client = instructor.from_mistral(client=client, mode=instructor.Mode.MISTRAL_TOOLS)

    try:
        repertoire = patched_client.chat.completions.create(
            model="mistral-large-latest",
            response_model = Repertoire,
            messages=[
                {
                    "role": "system",
                    "content": SYSTEM_PROMPT,
                },
                {
                    "role": "user",
                    "content": f"I have added a feature that forces you to response only in `locale={language}` and consider only chilean spanish.",
                },
                {
                    "role": "assistant",
                    "content": f"Understood thank you. From now I will only response with `locale={language}`",
                },
                {
                    "role": "user",
                    "content": txt,
                },
                {"role": "user", "content": USER_PROMPT},
            ],
            temperature=0.7,
            top_p=1,
        )
    except Exception as e:
        logger.info(f"Error creating jokes from transcript. {e}")
        repertoire = Repertoire(jokes=[])

    return repertoire