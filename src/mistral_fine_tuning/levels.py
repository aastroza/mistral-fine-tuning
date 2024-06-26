from openai import AsyncOpenAI
from mistralai.async_client import MistralAsyncClient
import instructor
from pydantic import BaseModel, Field
from typing import List

from dotenv import load_dotenv
load_dotenv()

client = instructor.patch(AsyncOpenAI())

mistral_client = MistralAsyncClient()
patched_client = instructor.from_mistral(client=mistral_client, mode=instructor.Mode.MISTRAL_TOOLS)

class Node(BaseModel):
    id: int
    label: str

class Edge(BaseModel):
    source: int
    target: int
    label: str

class KnowledgeGraph(BaseModel):
    nodes: List[Node] = Field(..., default_factory=list)
    edges: List[Edge] = Field(..., default_factory=list)

SYSTEM_PROMPT = """
You are tasked with analyzing jokes to identify and visualize the intentionality or mindstates involved using knowledge graphs.

Because mentalising underpins our ability to handle metaphor in conversations, and because jokes depend heavily on 
metaphorical use of language, we had wondered how mentalising ability affected our appreciation of jokes.
We found that most jokes consisted of either three or five mindstates 
(counting the mindstates of the audience member and the comedian as two of these), 
with only a handful having six or seven mindstates. 

To give you a flavour of these, here are two examples, one a second order joke 
(there are no mindstates in it other than the comedian’s and the audience’s – you) 
from the American comedian George Wallace, the other a fifth order joke that has been recycled 
so often no one seems to know who originally created it. 

Second order joke: At the airport they asked me if anybody I didn’t know gave me anything. Even the people I know don’t give me anything. 
Knowledge graph:
```json
{
    "nodes": [
        {
            "id": 1,
            "label": "Comedian",
        },
        {
            "id": 2,
            "label": "Audience",
        }
    ],
    "edges": [
        {
            "source": 1,
            "target": 2,
            "label": "entertains",
        },
        {
            "source": 2,
            "target": 1,
            "label": "understands joke",
        }
    ]
}
```

Fifth order joke: A young boy enters a barber shop and the barber whispers to his customer, ‘This is the dumbest kid in the world. Watch while I prove it to you.’ The barber puts a dollar bill in one hand and two quarters in the other, then calls the boy over and asks, ‘Which do you want, son?’ The boy takes the quarters and leaves. ‘What did I tell you?’ said the barber. ‘That kid never learns!’ Later, when the customer leaves, he sees the same young boy coming out of the ice cream store. ‘Hey, son! May I ask you a question? Why did you take the quarters instead of the dollar bill?’ The boy licked his cone and replied, ‘Because the day I take the dollar, the game is over!”
Knowledge graph:
```json
{
    "nodes": [
        {
            "id": 1,
            "label": "Comedian",
        },
        {
            "id": 2,
            "label": "Barber",
        },
        {
            "id": 3,
            "label": "Customer",
        },
        {
            "id": 4,
            "label": "Boy",
        },
        {
            "id": 5,
            "label": "Audience",
        }
    ],
    "edges": [
        {
            "source": 1,
            "target": 5,
            "label": "entertains",
        },
        {
            "source": 2,
            "target": 4,
            "label": "sets up joke with",
        },
        {
            "source": 3,
            "target": 2,
            "label": "observes",
        },
        {
            "source": 4,
            "target": 2,
            "label": "outsmarts",
        },
        {
            "source": 5,
            "target": 2,
            "label": "understands",
        },
        {
            "source": 5,
            "target": 3,
            "label": "understands",
        },
        {
            "source": 5,
            "target": 4,
            "label": "understands",
        }
    ]
}
```
"""

async def extract_graph(text: str, language: str) -> KnowledgeGraph:
    return await client.chat.completions.create(
        model="gpt-4o",
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
             "content": f"Help me understand the following joke by describing it as a detailed knowledge graph of mindstates: {text}"
             },
        ],
        temperature=0.8,
        response_model=KnowledgeGraph,
    )

async def extract_graph_mistral(text: str, language: str) -> KnowledgeGraph:
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
             "content": f"Help me understand the following joke by describing it as a detailed knowledge graph of mindstates: {text}"
             },
        ],
        temperature=0.8,
        response_model=KnowledgeGraph,
    )
