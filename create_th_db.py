"""
This script creates a thematic database out of a list of string documents.

Documents should usually be about paragraph-sized for optimal results.

A thematic deconstruction of a document is a list of the events described in it,
each represented as a dictionary from thematic roles to their content (the entities occupying them).

Thematic roles are the roles that participants in a sentence take on in relation to the action or state described by the sentence's verb.

They are distinct from syntactic concepts like subject and object because they describe the semantic contribution of a certain phrase to the sentence.

This deconstruction is done using ChatGPT-4 with the prompts/system_thematic.jinja2 prompt.

An example output:

[{'Agent': 'wireless market', 'Manner': 'highly', 'Predicate': 'compete'},
 {'Agent': 'our connections', 'Goal': 'customers', 'Predicate': 'grow'},
 {'Agent': 'churn rates',
  'Goal': 'our network superiority and reliability',
  'Patient': 'customers',
  'Predicate': 'signal'},
 {'Agent': 'our network superiority and reliability',
  'Patient': 'customers',
  'Predicate': 'attract'},
 {'Agent': 'we', 'Patient': 'unique product', 'Predicate': 'offer'},
 {'Agent': 'we', 'Manner': 'excited', 'Predicate': 'enter', 'Time': '2019'},
 {'Agent': '5G', 'Goal': 'possibilities', 'Predicate': 'bring'},
 {'Agent': 'customers',
  'Patient': 'best products, best plans, best network',
  'Predicate': 'benefit'}]

This represents the document:

“In a highly competitive wireless market, our connections growth and churn rates signal that customers are attracted to our network superiority and reliability, as well as our unique product offerings,” Dunne said. “We enter 2019 excited about the possibilities that 5G will bring, and confident that customers will benefit from the best products, on the best plans, on the best network.”

To query ChatGPT using the load balancer object as we do here, you would need OpenAI and Azure API key, as well as an Azure API Base.
"""
import json

import lbgpt
import diskcache
import pandas as pd
import os

import jinja2
import asyncio
from datasets import load_dataset

OPENAI_API_KEY = 'YOUR_OPENAI_API_KEY'
AZURE_OPENAI_API_KEY = 'YOUR_AZURE_OPENAI_API_KEY'
AZURE_OPENAI_API_BASE = 'YOUR_AZURE_OPENAI_API_BASE'

# A jinja2 environment to render the prompt
temp_env = jinja2.Environment(
    loader=jinja2.FileSystemLoader('prompts'),
    autoescape=jinja2.select_autoescape(),
)

# This object is used to query chatgpt more efficiently
cache = diskcache.Cache('cache')
chatgpt = lbgpt.LoadBalancedGPT(
    openai_api_key=OPENAI_API_KEY,
    azure_api_key=AZURE_OPENAI_API_KEY,
    azure_api_base=AZURE_OPENAI_API_BASE,
    cache=cache,
    azure_model_map={
        "gpt-3.5-turbo-0613": "gpt-35-turbo-0613",
        "gpt-3.5-turbo": "gpt-35-turbo-0613",
        "gpt-4": "gpt-4-0613",
        "gpt-4-0613": "gpt-4-0613",
    },
)


def thematize(docs):
    """
    Converts a list of documents into a list of lists of event dictionaries
    as described in the preamble.
    """

    # Create a list of queries, one for each doc
    comp_list = []
    for doc in docs:
        d = {
            # Use the jinja2 environment to render the prompt
            # both from the user side (the argument, which is doc),
            # and the system side, which is the instruction to chatgpt.
            'messages': [
                {'role': 'system',
                 'content':
                     temp_env.get_template('system_thematic.jinja2').render()},
                {'role': 'user',
                 'content': temp_env.get_template('user_thematic.jinja2').render(doc=doc)}
            ],

            # Model parameters
            'model': 'gpt-4', 'temperature': 0, 'max_tokens': 2000, 'top_p': 1, 'frequency_penalty': 0,
            'presence_penalty': 0, 'request_timeout': 500,
        }

        comp_list.append(d)

    # Perform the querrying using the LoadBalancedGPT object
    outs = asyncio.run(chatgpt.chat_completion_list(comp_list))
    return [out.choices[0].message['content'] for out in outs]


# Load the documents you would like to use
# Here we use 100 documents from the AG News dataset as an example
dataset_name = 'ag_news'
dataset = load_dataset(dataset_name, split="train")
docs = dataset.shuffle().select(range(100))['text']

# Create the database
db = thematize(docs)

# Save database
os.makedirs('output', exist_ok=True)
source2db = {c: r for c, r in zip(docs, db)}
json.dump(source2db, open(f'output/{dataset_name}.json', 'w'))
