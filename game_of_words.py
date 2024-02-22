import os
import glob
import openai
import random
from transformers import pipeline
from tqdm import tqdm
import torch
from pprint import pprint
import jinja2
from collections import defaultdict
import re
import requests

openai.api_key = 'your_openai_key'

response_prompt_url = "https://github.com/Marvin-Labs/ml-examples/blob/main/prompts/conv_social.txt"
pick_qud_prompt_url = "https://github.com/Marvin-Labs/ml-examples/blob/main/prompts/pick_qud.txt"
Tyrion_url = "https://github.com/Marvin-Labs/ml-examples/blob/main/data/people/Tyrion.txt"
Eddard_url = "https://github.com/Marvin-Labs/ml-examples/blob/main/data/people/Eddard.txt"


def read_text_file(url):
    try:
        # Send a GET request to fetch the contents of the URL
        response = requests.get(url)

        # Check if the request was successful (status code 200)
        if response.status_code == 200:
            # Check if the content type is text/plain
            if 'text/plain' in response.headers.get('content-type', ''):
                # Read the content of the text file
                content = '\n'.join(
                    eval(response.text.replace('false', 'False').replace('true', 'True').replace('null', 'None'))[
                        'payload']['blob']['rawLines'])
                return content
            else:
                print("The URL does not point to a text file.")
                return None
        else:
            print("Failed to fetch the URL. Status code:", response.status_code)
            return None
    except Exception as e:
        print("An error occurred:", e)
        return None


def tokenize_text(text):
    # Split the text into alphanumeric words and punctuation
    tokens = re.findall(r'\w+|[^\w\s]', text)
    return tokens


# Function to query ChatGPT
def query_chatgpt(prompt, functions=[]):
    """
    Queries the ChatGPT model with a given prompt.

    Args:
    - prompt (str): The prompt text to send to the model.
    - functions (list, optional): Additional functions for the model to use.

    Returns:
    - str: The content of the response from the model.
    """
    # Set up initial system message
    messages = [{"role": "system", "content": prompt}]

    # Define model parameters
    model_params = {
        "messages": messages,
        "model": "gpt-4",
        "temperature": 0,
        "max_tokens": 100,
        "top_p": 1,
        "frequency_penalty": 0,
        "presence_penalty": 0
    }

    # Include additional functions if provided
    if functions:
        model_params['functions'] = functions

    # Get response from the model
    response = openai.ChatCompletion.create(**model_params)
    return response['choices'][0]['message']['content']


def game_of_words():
    # These values should be parametrized in the function if you want to use different agents with different KBs
    agent1 = "Tyrion"
    agent2 = "Eddard"
    kb1 = read_text_file(Eddard_url)
    kb2 = read_text_file(Tyrion_url)

    # If we pick the first question policy, we also add this clause to further restrict questions
    question_restriction_clause = "\n\n Important: DO NOT ask questions if you have something interesting or relevant to contribute about the current QUD!!!!!!!!\nQuestions should be used strategically and prudently."

    qud_instructions = read_text_file(pick_qud_prompt_url)
    response_instructions = read_text_file(response_prompt_url)

    # We have two question policies that we randomize between
    question_policy1 = "Your response should include a question only if you feel like you've dwelt on the same topic for a while or you <OTHER_NAME> said something that's surprising given your KB and you want more info."
    question_policy2 = ""

    # QUD picking strategy to randomize
    # Only relevant if the agent hasn't chosen 'keep' or 'address'
    strategies = ['abstract (Find a generalization of the QUD)',
                  'specify (Ask about a specific subcase of the QUD)',
                  'contrast (Ask a question about some counterexample)',
                  'disagree (Ask a question about a flaw)']

    # This just to kickstart the conversation
    cg = 'Eddard: You are the oldest child in your family, Tyrion. Don\'t you agree?'
    qud1 = 'Lannister Family'
    qud2 = 'Lannister Family'

    while True:

        # Randomize a probability for question policy
        question_p = random.random()

        # Randomize strategy and update QUD prompt accordingly
        strat1 = random.choice(strategies)
        qud_prompt1 = qud_instructions.replace('<KB>', kb1).replace('<QUD>', qud1).replace('<CG>', cg).replace(
            '<SELF_NAME>', agent1).replace('<OTHER_NAME>', agent2).replace('<STRATEGY>', strat1)

        # Get QUD
        strat_andor_qud1 = query_chatgpt(qud_prompt1)

        # Just some string manipulation
        if '|' in strat_andor_qud1:
            strat1, qud1 = strat_andor_qud1.split('|')
        elif strat_andor_qud1 == '<keep>':
            pass
        else:
            qud1 = strat_andor_qud1

        strat1 = strat1.strip()
        qud1 = qud1.strip()

        # Here we print the strategy chosen and the QUD
        print('SQ1:', ' | '.join([qud1, strat1]))

        # Prepare the response prompt
        res_prompt1 = response_instructions.replace('<KB>', kb1).replace('<QUD>', qud1).replace('<CG>', cg).replace(
            '<SELF_NAME>', agent1).replace('<OTHER_NAME>', agent2)

        # Update it according to the question probability randomized
        if question_p <= .3:
            res_prompt1 = res_prompt1.format(question_policy1)
            res_prompt1 += question_restriction_clause
        else:
            res_prompt1 += question_policy2

        # Get response and parse
        res1 = query_chatgpt(res_prompt1)
        out1 = agent1 + ': ' + res1
        print(out1)

        # Update CG
        cg += out1 + '\n'

        # Same for agent2 (I know I should eventually put this in a function)

        strat2 = random.choice(strategies)
        qud_prompt2 = qud_instructions.replace('<KB>', kb2).replace('<QUD>', qud2).replace('<CG>', cg).replace(
            '<SELF_NAME>', agent2).replace('<OTHER_NAME>', agent1).replace('<STRATEGY>', strat2)

        strat_andor_qud2 = query_chatgpt(qud_prompt2)
        if '|' in strat_andor_qud2:
            strat2, qud2 = strat_andor_qud2.split('|')
        elif strat_andor_qud2 == '<keep>':
            pass
        else:
            qud2 = strat_andor_qud2

        strat2 = strat2.strip()
        qud2 = qud2.strip()

        print('SQ2:', ' | '.join([qud2, strat2]))

        res_prompt2 = response_instructions.replace('<KB>', kb2).replace('<QUD>', qud2).replace('<CG>', cg).replace(
            '<SELF_NAME>', agent2).replace('<OTHER_NAME>', agent1)

        if .3 < question_p < 0.6:
            res_prompt2 = res_prompt2.format(question_policy1)
            res_prompt2 += question_restriction_clause
        else:
            res_prompt2 += question_policy2

        res2 = query_chatgpt(res_prompt2)
        out2 = agent2 + ': ' + res2
        print(out2)
        cg += out2 + '\n'


game_of_words()