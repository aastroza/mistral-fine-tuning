# Mistral AI Fine-tuning Hackaton

## Summary

**[[Video]](https://www.youtube.com/watch?v=XbIbgFgQVmQ)**

Generative AI is revolutionizing communication worldwide. However, current models are [predominantly trained in standard English](https://blog.modernmt.com/making-generative-ai-multilingual-at-scale/), neglecting other languages like the unique Spanish dialect spoken in Chile. This dialect is considered [highly disruptive globally](https://www.elmundo.es/cultura/2021/11/30/61a4a36321efa013518b4571.html) due to its rapid evolution in spoken and written language, creation of new words, and flexible rules of pronunciation. Adapting AI models to this Chilean dialect poses a challenge, requiring a curated database reflecting the language's evolution for accuracy.

Chile boasts a rich tradition in humor, showcased annually at the [Viña del Mar Festival](https://en.wikipedia.org/wiki/Vi%C3%B1a_del_Mar_International_Song_Festival) since 1960. Leveraging the [Mistral API](https://docs.mistral.ai/), we created a database with more than 4,000 jokes from the festival, used to fine-tune a model generating jokes based on keywords. An additional step involved [mapping the mentalization levels in joke structure](https://pubmed.ncbi.nlm.nih.gov/26597196/) for quality control.

A [website](https://www.datarisas.cl/) was developed to compare our model against others by presenting pairs of jokes for user preference assessment. Based on votes from hundreds of Chileans, our model currently leads in the ELO ranking.

## How to run the code

Run the following Jupyter notebooks:

1. [Extract jokes from transcripts](/notebooks/01_extract_jokes_from_transcripts.ipynb)
2. [Refine jokes to build a fine-tuning dataset](/notebooks/02_refine_jokes_dataset.ipynb)
3. [Prepare the fine-tuning dataset](/notebooks/03_prepare_dataset.ipynb)
4. [Build baselines, fine-tune and evaluation](/notebooks/04_building_baselines_and_fine_tuning.ipynb)

## Documentation

### Motivation

### Extracting jokes from Youtube transcripts

### Generating a dataset for LLM fine-tuning

### Fine-tuning the models

### Model evaluation

### Future Work