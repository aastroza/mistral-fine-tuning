# Mistral AI Fine-tuning Hackaton

## Summary

**[[Video]](https://youtu.be/Wb4J4xOhjGo?si=k-zOEQCmSK5ucvBM)**

Generative AI is revolutionizing communication worldwide. However, current models are [predominantly trained in standard English](https://blog.modernmt.com/making-generative-ai-multilingual-at-scale/), neglecting other languages like the unique Spanish dialect spoken in Chile. This dialect is considered [highly disruptive globally](https://www.elmundo.es/cultura/2021/11/30/61a4a36321efa013518b4571.html) due to its rapid evolution in spoken and written language, creation of new words, and flexible rules of pronunciation. Adapting AI models to this Chilean dialect poses a challenge, requiring a curated database reflecting the language's evolution for accuracy.

Chile boasts a rich tradition in humor, showcased annually at the [Viña del Mar Festival](https://en.wikipedia.org/wiki/Vi%C3%B1a_del_Mar_International_Song_Festival) since 1960. Leveraging the [Mistral API](https://docs.mistral.ai/), we created a database with more than 4,800 jokes from the festival and used a high quality subset to fine-tune a model for generating jokes based on keywords. An additional step involved [mapping the mentalization levels in joke structure](https://pubmed.ncbi.nlm.nih.gov/26597196/) for quality control.

We compared Mistral 7B (non-finetuned) with our Mistral 7B finetuned model, using Mistral Large as a judge. The score improved from 43.12% to 49.09%.

## How to run the code

Run the following Jupyter notebooks:

1. [Extract jokes from transcripts](/notebooks/01_extract_jokes_from_transcripts.ipynb)
2. [Refine jokes to build a fine-tuning dataset](/notebooks/02_refine_jokes_dataset.ipynb)
3. [Prepare the fine-tuning dataset](/notebooks/03_prepare_dataset.ipynb)
4. [Build baselines, fine-tune and evaluation](/notebooks/04_building_baselines_and_fine_tuning.ipynb)

## Documentation

### Motivation

Generative AI is revolutionizing global communication, with the potential to mediate all our interactions in the near future. However, [current models predominantly excel in standard English](https://blog.modernmt.com/making-generative-ai-multilingual-at-scale/), leaving less-represented languages and dialects at a significant disadvantage. This disparity is particularly evident in the case of **Chilean Spanish**, a dialect so unique that it challenges the very notion of what constitutes Spanish.

The Spanish spoken in Chile [stands out as one of the most linguistically disruptive variants worldwide](https://www.elmundo.es/cultura/2021/11/30/61a4a36321efa013518b4571.html). Its distinctiveness is unmistakable to both native speakers and language experts, although it is difficult to quantify. The Chilean dialect is characterized by:

- Rapid evolution of spoken and written language
- Frequent invention and adoption of new words
- Flexible and often unconventional pronunciation
- Constant modification and reinterpretation of linguistic rules

These factors combine to create a language environment where *"speaking Chilean"* diverges significantly from standard Spanish, posing unique challenges for LLMs.

The legendary filmmaker [Raúl Ruiz](https://www.ojoentinta.com/chile-segun-raul-ruiz/) eloquently captures the complexity of Chilean Spanish:

>"What I like about Chile is that special way Chileans have of speaking. Chileans are sometimes capable of speaking without using either verb or subject, or they use verbs and the subject displaced, which makes them talk for hours and you don't know what about. Every Chilean speaks exclusively in quotation marks. It's someone who puts rhetoric before reality. Chile manufactures a very curious form of artificial language in which intonation is almost as important as the words that are uttered. More than the accent, it's the strange syntax. One starts a sentence and ends with ellipsis, starts another and another, and what happens is that people are speaking with three parallel discourses."

Adapting models to this Chilean dialect requires more than simple translation. It demands a deep understanding of the cultural context, linguistic nuances, and the ever-evolving nature of the language. To achieve this, we need to curate a reliable written record that captures the essence of Chilean Spanish, reflecting its evolution while ensuring quality and representativeness.

An unexpected but rich source for this linguistic data lies in Chile's vibrant comedy scene. The [Viña del Mar Festival](https://en.wikipedia.org/wiki/Vi%C3%B1a_del_Mar_International_Song_Festival), showcases the country's top comedians and offers a treasure trove of uniquely Chilean expressions, wordplay, and cultural references. Held annually since 1960, the festival presents Chilean humor, known for its eccentricity – from talking puppets to trampoline-jumping comedians. This humor often features dialogues that are incomprehensible to non-Chilean Spanish speakers, making it an ideal dataset for training LLMs in the intricacies of Chilean Spanish.

Developing a model specialized in Chilean Spanish is not just about preserving linguistic diversity. It's about ensuring that as AI-mediated communication becomes ubiquitous, Chilean voices are not left behind. This project aims to bridge the gap between global advancements and local linguistic realities. The goal is to create a LLM that can authentically understand and interact with Chilean Spanish speakers, reflecting their unique expressions and cultural context.

### Data

#### Extracting jokes from Youtube transcripts

```python
class Segment:
    start_time: float
    end_time: float
    transcript: str
```

```python
class Joke(BaseModel):
    transcript: str = Field(
        description="The joke transcript. Do not include comments, greetings, or any other non-joke content."
    )
    corrected_transcript: str = Field(
        description="The corrected joke transcript. Clean the transcript from any unnecessary content. Fix typos. Ensure correct use of punctuation. Make sure the joke is clean like a historical quote."
    )

class Repertoire(BaseModel):
    jokes: List[Joke]
```

#### Ensuring quality

```python
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
```

```python
class Quality(Enum):
    FUNNY = "funny"
    NOT_FUNNY = "not_funny"

class Refinement(BaseModel):
    text: str = Field(..., description="The corrected joke text. Fix grammar. Fix typos and missing characters. Ensure correct use of punctuation.")    
    keywords: List[str] = Field(..., default_factory=list, description="The corrected keywords array")
    quality: Quality = Field(..., description="The quality of the joke")

class Rewrite(BaseModel):
    rewritten_text: str = Field(..., description="The rewritten joke. Maintaining the original humorous style but improving its grammar, punctuation, and presentation.")
```

### Fine-tuning

#### Using Mistral API + Weave

#### Creating baselines

```python
class MistralModel(weave.Model):
    model: str
    temperature: float = 0.7
    
    @weave.op
    def create_messages(self, keyword:str):
        return create_messages(keyword)

    @weave.op
    async def predict(self, keyword:str):
        messages = self.create_messages(keyword)
        return await call_mistral(model=self.model, messages=messages)
```

#### Model evaluation

```python
class LLMJudge(weave.Model):
    model: str = "mistral-large-latest"
    
    @weave.op
    async def predict(self, keyword: str, mistral_7b: str, mistral_medium: str, text: str, **kwargs) -> dict:
        messages = [
            ChatMessage(
                role="user",
                content=(
                "You are a world class comedian and you are judging a joke competition in Chile."
                "You have to pick the best joke between two jokes written about a keyword."
                "Take into consideration the jokes were written in Chilean Spanish and a ground truth joke as a reference. \n"
                "Here is the keyword: {keyword}\n"
                "Here is the joke1: {mistral_7b}\n"
                "Here is the joke2: {mistral_medium}\n"
                "Ground truth joke: {joke}\n"
                "Return the name of the best_joke (or None if you think both are bad) and the reason in short JSON object.").format(
                    keyword=keyword, 
                    mistral_7b=mistral_7b, 
                    mistral_medium=mistral_medium,
                    joke=text)
            )
        ]
        payload = await call_mistral(model=self.model, messages=messages, response_format={"type": "json_object"})
        return json.loads(payload)
```

### Future Work


## Acknowledgments


- [**Mastering LLMs: A Conference For Developers & Data Scientists**](https://maven.com/parlance-labs/fine-tuning) by [Dan Becker](https://github.com/dansbecker) and [Hamel Husain](https://github.com/hamelsmu). This course provided an excellent introduction to fine-tuning and inspired my participation in the [Mistral AI fine-tuning hackathon](https://mistral.ai/news/2024-ft-hackathon/).
    - Special thanks to [Sophia Yang](https://github.com/sophiamyang) (Mistral AI) for her insights on the Mistral API, and [Thomas Capelle](https://github.com/tcapelle) (W&B) for his teachings on the impressive [Weave](https://wandb.github.io/weave/) toolkit. Much of the code and many ideas in this project were drawn from their invaluable lessons.

- [**Instructor**](https://github.com/jxnl/instructor): This tool was a significant help in our project. It allowed us to add structure to the output from the Mistral API, making it much easier to process our data and create a high-quality dataset for fine-tuning.

- We extend our gratitude to [R.I.M. Dunbar](https://en.wikipedia.org/wiki/Robin_Dunbar), [Jacques Launay](https://greatergood.berkeley.edu/profile/jacques_launay#:~:text=Jacques%20Launay%20is%20a%20Postdoctoral,at%20the%20University%20of%20Oxford.), and [Oliver Curry](https://www.oliverscottcurry.com/) for their insightful paper [**"The Complexity of Jokes Is Limited by Cognitive Constraints on Mentalizing"**](https://pubmed.ncbi.nlm.nih.gov/26597196/). Their research provided valuable parameters for defining good jokes, which we applied automatically thanks to LLMs. Their work on the relationship between joke complexity, levels of intentionality, and humor appreciation significantly improved our approach to generating and evaluating Chilean humor.