import asyncio
from mistralai.async_client import MistralAsyncClient
import instructor
from pydantic import BaseModel

from dotenv import load_dotenv
load_dotenv()

client = MistralAsyncClient()
patched_client = instructor.from_mistral(client=client, mode=instructor.Mode.MISTRAL_TOOLS)

class Person(BaseModel):
    name: str
    age: int

async def extract_person(text: str) -> Person:
    return await patched_client.chat.completions.create(
        model="mistral-large-latest",
        messages=[
            {"role": "user", "content": text},
        ],
        response_model=Person,
    )

async def main():

    dataset = [
        "My name is John and I am 20 years old",
        "My name is Mary and I am 21 years old",
        "My name is Bob and I am 22 years old",
        "My name is Alice and I am 23 years old",
        "My name is Jane and I am 24 years old",
        "My name is Joe and I am 25 years old",
        "My name is Jill and I am 26 years old",
    ]

    sem = asyncio.Semaphore(2)

    async def rate_limited_extract_person(text: str) -> Person:
        async with sem:
            return await extract_person(text)

    tasks_get_persons = [rate_limited_extract_person(text) for text in dataset]
    resp = await asyncio.gather(*tasks_get_persons)
    print("asyncio.gather (rate limited):", resp)

if __name__ == "__main__":
    asyncio.run(main())