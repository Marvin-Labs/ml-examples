campaign = 'GoW_data'
agent1_name = 'tyrion'
agent2_name = 'eddard'


import os
import pickle
import glob
import json
import openai
import random
from transformers import pipeline
import faiss
from tqdm import tqdm
import torch
from pprint import pprint
import jinja2
from collections import defaultdict
import re
import requests


max_history_length = 12000 # Calculated to be approximately the maximal length that wouldn't bypass the context window limitation
address_min_dist = 10


strat_dict = {
                'abstract': 'Find a generalization of the QUD.',
                'specify': 'Ask about a specific subcase of the QUD. You may consider a different domain.',
                'contrast': 'Ask a question about some counterexample. You may compare to a different domain.',
                'challenge': 'point out a possible point of contention',
                'exemplify': 'point out another topic/domain/person on which this QUD also applies',
}

named_strats = ['abstract (Find a generalization of the QUD)',
                  'specify (Ask about a specific subcase of the QUD)',
                  'contrast (Ask a question about some counterexample)',
                  'disagree (Ask a question about a flaw)']

strats = list(strat_dict.keys())


question_policy_unasky = "Your response should include a question only if you feel like you've dwelt on the same topic for a while or you <OTHER_NAME> said something that's surprising given your KB and you want more info."
question_policy_asky = "or asks for such information. You may ask for additional relevant information if you need it to resolve the QUD."

question_restriction_clause = "\n\n Important: DO NOT ask questions if you have something interesting or relevant to contribute about the current QUD!!!!!!!!\nQuestions should be used strategically and prudently."


def read_text_file(url):
    try:
        # Send a GET request to fetch the contents of the URL
        response = requests.get(url)

        # Check if the request was successful (status code 200)
        if response.status_code == 200:
            # Check if the content type is text/plain
            if 'text/plain' in response.headers.get('content-type', ''):
                # Read the content of the text file
                content = '\n'.join(eval(response.text.replace('false', 'False').replace('true', 'True').replace('null', 'None'))['payload']['blob']['rawLines'])
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

def topic_kb_entry_string(k,v):
    factoid_string = '\n-'.join(v)
    return f'{k}:\n\n-{factoid_string}'

def tokenize_text(text):
    # Split the text into alphanumeric words and punctuation
    tokens = re.findall(r'\w+|[^\w\s]', text)
    return tokens


# Function to query ChatGPT
def query_chatgpt(prompt, max_tokens=100, functions=[]):
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
        "max_tokens": max_tokens,
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



response_prompt_url = "https://github.com/Marvin-Labs/ml-examples/blob/main/prompts/conv_social.txt"
pick_qud_prompt_url = "https://github.com/Marvin-Labs/ml-examples/blob/main/prompts/pick_qud.txt"

approach_instructions = open(f'{campaign}/prompts/pick_approach.txt').read()
response_instructions = open(f'{campaign}/prompts/conv_social.txt').read()
transform_qud_instructions = open(f'{campaign}/prompts/transform_qud.txt').read()
deduce_instructions = open(f'{campaign}/prompts/deduce_implications.txt').read()
refresh_instructions = open(f'{campaign}/prompts/refresh_conv.txt').read()

# This for running the respond prompt from Episode I
original_response_instructions = open(f'{campaign}/prompts/respond.txt').read()
original_response_instructions = read_text_file(response_prompt_url)

qud_instructions = read_text_file(pick_qud_prompt_url)

class Conversation:
    pass

class Duologue(Conversation):

    def __init__(self, agent1_name, agent2_name,
                 max_n_auds=3):
        self.agent1 = Agent(agent1_name, max_n_auds=max_n_auds)
        self.agent2 = Agent(agent2_name, max_n_auds=max_n_auds)

        sorted_names = sorted([self.agent1.name, self.agent2.name])
        self.id = f'{sorted_names[0]}+{sorted_names[1]}'

        self.global_cg_path = f'{campaign}/seeds/cgs/global.json'
        if os.path.exists(self.global_cg_path):
            self.cg = eval(open(self.global_cg_path).read())
        else:
            self.cg = {}

        self.cg_path = f'{campaign}/db/cgs/{self.id}.json'
        if os.path.exists(self.cg_path):
            particular_cg = eval(open(self.cg_path).read())
        else:
            particular_cg = {}

        self.cg.update(particular_cg)

        self.log_path = f'{campaign}/db/logs/{self.id}.txt'
        os.makedirs(os.path.dirname(self.log_path), exist_ok = True)
        if not os.path.exists(self.log_path):
            open(self.log_path, 'w').close()

        history_path = f'{campaign}/db/logs/{self.id}_plain.txt'
        # if os.path.exists(history_path):
        #     self.history = [l.strip() for l in open(history_path).readlines()]
        #     init_res_named = self.history[-1]
        # else:
        #     init_res_named = f"{self.agent2.name.capitalize()}: tell me whats on your mind."
        #     self.history = [init_res_named]

        self.history = []
        # topic = self.agent2.pick_topic_from_stm(self.cg, self.agent1.name)
        # res2 = 'Tell me about ' + topic
        topic = f"how is {self.agent1.name} doing"
        self.agent1.qud = self.agent2.qud = topic
        res2 = f"{self.agent2.name}: Tell me what's on your mind, {self.agent1.name}."
        # res_prompt = response_instructions.replace('<KB>', self.agent2.kb).replace('<QUD>', topic).replace('<CG>', '\n'.join(self.history)[-max_history_length:]).replace(
        #     '<SELF_NAME>', self.agent2.name).replace('<OTHER_NAME>', self.agent1.name).replace('<PERSONALITY>', self.agent2.personality).replace('<PREVIOUS_MESSAGE>', init_res_named)
        #
        # res2 = query_chatgpt(res_prompt)

        # self.history.append(res1)

        # res2 = self.agent2.respond(self.agent1.name, '', res1, True, self.cg, self.log_path)
        res2_named = f"{self.agent2.name.capitalize()}: {res2}"
        self.history.append(res2_named)
        self.agent2.print(res2_named)

        # thats a simpler way to do it:
        #res2 = f'Hello, {self.agent1.name}!'
        #self.history = [f"{self.agent2.name}: {res2}"]


    def run(self):

        res2 = self.history[-1]
        try:
            while True:
                question_p = random.random()

                res1 = self.agent1.respond(self.agent2.name, '\n'.join(self.history), res2, question_p < .15, self.cg, self.log_path)
                res1_named = f"{self.agent1.name.capitalize()}: {res1}"
                self.history.append(res1_named)

                res2 = self.agent2.respond(self.agent1.name, '\n'.join(self.history), res1, .15 < question_p < .3, self.cg, self.log_path)
                res2_named = f"{self.agent2.name.capitalize()}: {res2}"
                self.history.append(res2_named)
                self.save()

        except KeyboardInterrupt:
            self.agent1.save()
            self.agent2.save()
            self.save()

    def save(self):

        log_path = f'{campaign}/db/logs/{self.id}_plain.txt'
        os.makedirs(os.path.dirname(log_path), exist_ok=True)
        with open(log_path, 'w') as f:
            f.write('\n'.join(self.history))

        # TODO: generate/update shared kb with topic modeling

    def load(self):
        # TODO: load agents and history
        pass


"Render a scene of two unlikely companions, Tyrion Lannister and Eddard Stark, engaged in a friendly banter. Tyrion, a smart and cynical medieval midget dwarf at 30 years old, dons a red outfit with his brown hair cascading down his shoulders while Eddard, a dutiful and honorable lord at 35 years old, stands tall in his white attire on top of the fortifications of the Red Keep at king's Landing."

class Agent:

    def __init__(self, name, from_seed=True, keep_patience=5, max_n_auds=3, max_n_deductions=2):
        self.name = name
        self.log_path = None
        self.seed_kb_path = f'{campaign}/seeds/kbs/{self.name}.txt'
        self.seed_personality_path = f'{campaign}/seeds/personalities/{self.name}.txt'

        self.dyn_kb_path = f'{campaign}/db/kbs/{self.name}.txt'
        self.dyn_personality_path = f'{campaign}/seeds/personalities/{self.name}.txt'

        try:
            self.personality = open(self.seed_personality_path).read()

            self.kb = self.personality
            self.kb += open(f'{campaign}/seeds/kbs/global.txt').read()
            self.kb += open(self.seed_kb_path, encoding='utf8').read()

        except FileNotFoundError:
            raise FileNotFoundError(f'KB or Personality file not found for {self.name}')

        if os.path.exists(self.dyn_kb_path):
            self.kb += open(self.dyn_kb_path, 'r').read()

        self.max_n_auds = max_n_auds
        self.keep_patience = keep_patience
        self.max_n_deductions = max_n_deductions

        self.qud_waitlist =  []
        self.auds = []
        self.qud = ''
        self.has_deduced = False
        self.just_changed_topic = False
        self.circling_back = False
        self.addressing = False
        self.keep_counter = 0
        self.deduce_counter = 0
        self.non_address_counter = address_min_dist

    def respond_simple(self, other_name, history, other_msg, asky, shared_kb={}, log_path=None):
        """
        This corresponds to responding from Episode I.
        """
        strat = random.choice(named_strats)

        history_curtailed = history[-max_history_length:]

        qud_prompt = qud_instructions.replace('<KB>', self.kb).replace('<QUD>', self.qud).replace('<CG>', history_curtailed).replace(
            '<SELF_NAME>', self.name).replace('<OTHER_NAME>', other_name).replace('<STRATEGY>', strat)

        # Get QUD
        approach_andor_qud = query_chatgpt(qud_prompt)

        # Just some string manipulation
        if '|' in approach_andor_qud:
            approach, self.qud = approach_andor_qud.split('|')
        elif approach_andor_qud == '<keep>':
            approach = strat
        else:
            self.qud = approach_andor_qud
            approach = strat

        approach = approach.strip()
        self.qud = self.qud.strip()

        # Here we print the strategy chosen and the QUD
        self.print('SQ:' + ' | '.join([self.qud, approach]))

        # Prepare the response prompt
        res_prompt = original_response_instructions
        res_prompt = res_prompt.replace('<KB>', self.kb).replace('<QUD>', self.qud).replace('<CG>', history_curtailed)\
            .replace('<SELF_NAME>', self.name).replace('<OTHER_NAME>', other_name)

        if asky:
            res_prompt = res_prompt.format(question_policy_asky)
        else:
            res_prompt = res_prompt.format(question_policy_unasky)
            res_prompt += question_restriction_clause

        # words = [w.strip() for w in open(f'{campaign}/prompts/words.txt').readlines()]
        # word_sample = ', '.join(random.sample(words, 10))
        # res_prompt += "\n\nIn your response, focus on these concepts: " + word_sample

        # Get response and parse
        with open('chronicles/db/logs/saga+zephyr.txt', 'a') as f:
            print(res_prompt, file=f)

        res = query_chatgpt(res_prompt, max_tokens=500)

        self.print(self.name.capitalize() + ': ' + res)
        self.print('')

        return res



    def respond(self, other_name, history, other_msg, asky, shared_kb={}, log_path=None):
        """
        This is the implementation of the pseudocode from Episode II.
        """

        #TODO: if they come up with an action item add it to the agenda

        # This is just an abbreviation. In reality you should pass log_path into self.print every time you call it
        self.log_path = log_path

        # TODO: Find a way to tie together different topics
        #(one way is to add a strat that ties topics together)
        # TODO: figure out the thing about CG = history + shared_kb
        # TODO: draw connections to previous other topics
        approach_prompt = approach_instructions.replace('<KB>', self.kb).replace('<QUD>', self.qud).replace('<CG>', history[-max_history_length:]).replace('<SELF_NAME>', self.name).replace('<OTHER_NAME>', other_name)

        if self.keep_counter >= self.keep_patience:
            approach_prompt = approach_prompt\
                .replace('<KEEP_POLICY1>', 'suggest a new QUD.')\
                .replace('<KEEP_POLICY2>', '')
            self.keep_counter = 0
        else:
            approach_prompt = approach_prompt\
                .replace('<KEEP_POLICY1>', 'decide whether to maintain the current QUD or switch it')\
                .replace('<KEEP_POLICY2>', 'If more discussion on the current QUD is warranted without a clear resolution from the conversation history, return <keep>.')

        if self.non_address_counter < address_min_dist:
            approach_prompt = approach_prompt.replace("includes a particularly surprising or unexpected fact from another topic that you just have to comment on",
                                                      "is a full quote of the entirety of the Hebrew Bible"
                                                      )
        else:
            self.non_address_counter = 0
        approach_output = query_chatgpt(approach_prompt, max_tokens=500).strip()

        if '<address>' not in approach_output:
            self.non_address_counter += 1

        if '<keep>' not in approach_output:
            self.print(approach_output)
            print('#AUDs & WL', len(self.auds), len(self.qud_waitlist))

        if approach_output == '<keep>':
            self.keep_counter += 1

        else:

            approach, B = [x.strip() for x in approach_output.split('|')]

            if approach == '<address>':
                if len(self.qud_waitlist) < 3:
                    self.qud_waitlist.append(self.qud)
                self.qud = B
                self.addressing = True

            # Add up to 3 auds
            # Otherwise, pick a qud from the waitlist
            # Depending on randomness/relevance,

            # experimental
            elif approach == '<switch>':
                self.qud = B
                # Commit info

            else:

                assert approach == '<resolved>'

                # Commit up to 3 auds
                if len(self.auds) < self.max_n_auds:
                    if B:
                        self.auds.append(B)

                    # Commit AUD to KB. Corresponds to step 2a in the pseudocode
                    self.kb += '\n'+ B
                    # TODO: also to shared KB

                # Then, if auds haven't capped,
                # randomly decide if to pick the QUD transformation
                # or pick from the waitlist (unless empty)

                # TODO: instead of deciding just by the length of auds,
                # also take into account how relevant is the transformation

                if len(self.auds) < self.max_n_auds and ((not self.qud_waitlist) or random.random() < .7):

                    strat = random.choice(strats)
                    strat = strat + f'{(strat_dict[strat])}'

                    # TODO: the more interesting the topic is for the agent is, the closer the transform is to the original topic

                    transform_prompt = transform_qud_instructions.replace('<KB>', self.kb).replace('<QUD>', self.qud).replace(
                        '<CG>', history[-max_history_length:]).replace('<KB>', self.kb).replace('<SELF_NAME>', self.name).replace('<OTHER_NAME>', other_name).replace('<STRATEGY>',strat)

                    new_qud = query_chatgpt(transform_prompt).strip()
                    self.qud = new_qud

                # If the current QUD has been resolved and there are unresolved topics in the waitlist,
                # pick one of them
                # corresponds to step 2b
                elif self.qud_waitlist:


                    self.print('\nCircling back...\n')

                    self.qud = random.choice(self.qud_waitlist)
                    self.qud_waitlist.remove(self.qud)
                    self.circling_back = True

                # Step 2c
                elif self.deduce_counter < self.max_n_deductions:

                    self.print('\n\nDeducing implications...\n\n')

                    # See what follows from the AUDs + KB

                    self.qud = self.deduce_implications(shared_kb, other_name)
                    self.deduce_counter += 1

                else:

                    self.deduce_counter = 0

                    # Step 2d
                    self.qud = self.pick_topic_from_stm(shared_kb, other_name)
                    self.has_deduced = False
                    self.just_changed_topic = True

        if '<address>' not in approach_output:
            self.print('QUD: ' + self.qud)
        res_prompt = response_instructions.replace('<KB>', self.kb).replace('<QUD>', self.qud).replace('<CG>', history[-max_history_length:]).replace(
            '<SELF_NAME>', self.name).replace('<OTHER_NAME>', other_name).replace('<PERSONALITY>', self.personality).replace('<PREVIOUS_MESSAGE>', other_msg)

        if asky:
            res_prompt = res_prompt.format(question_policy_asky)
        else:
            res_prompt = res_prompt.format(question_policy_unasky)
            res_prompt += question_restriction_clause

        if self.just_changed_topic:
            res_prompt += '\n\nPhrase your response in a way of changing topic.'
            self.just_changed_topic = False
        elif self.circling_back:
            res_prompt += "\n\nPhrase your response in a way of circling back to an earlier topic in the conversation, with an expression like 'anyway', 'be that as it may', 'one way or the other', or whatever fits the style and context. Give a reason why you are circling back to that topic."
            self.circling_back = False
        elif self.addressing:
            res_prompt += f"\n\nPhrase your response in a way of addressing a noteworthy/surprising aspect of {other_name}'s message, along the lines of 'wait a second, you're playign tennis?' (but in a style that fits the context)."
            self.addressing = False

        res = query_chatgpt(res_prompt)

        self.print('')
        self.print(self.name.capitalize() + ': ' + res)
        self.print('')

        return res

    def deduce_implications(self, shared_kb, other_name):

        skb_as_str = '\n\n'.join([topic_kb_entry_string(k, v) for k, v in shared_kb.items()])

        deduce_prompt = deduce_instructions.replace('<KB>', self.kb) \
            .replace('<SELF_NAME>', self.name) \
            .replace('<OTHER_NAME>', other_name) \
            .replace('<AUDs>', '-' + '\n-'.join(self.auds)) \
            .replace('<SHARED_TOPICS>', skb_as_str)

        deduce_output = query_chatgpt(deduce_prompt).strip()

        # TODO: commit AUDs to KB

        return deduce_output


    def pick_topic_from_stm(self, shared_kb, other_name):

        self.print('\n\nPicking a new topic...\n\n')

        skb_as_str = '\n\n'.join([topic_kb_entry_string(k,v) for k,v in shared_kb.items()])

        refresh_prompt = refresh_instructions.replace('<KB>', self.kb) \
            .replace('<SELF_NAME>', self.name) \
            .replace('<OTHER_NAME>', other_name) \
            .replace('<AUDs>', '-' + '\n-'.join(self.auds)) \
            .replace('<SHARED_TOPICS>', skb_as_str)

        refresh_output = query_chatgpt(refresh_prompt).strip()
        
        return refresh_output

    def print(self, msg):

        print(msg)
        if self.log_path is not None:
            with open(self.log_path, 'a') as f:
                print(msg, file=f)

    def save(self):

        # Save kb separately
        os.makedirs(os.path.dirname(self.dyn_kb_path), exist_ok=True)
        if not os.path.exists(self.dyn_kb_path):
            open(self.dyn_kb_path, 'w').close()
        with open(self.dyn_kb_path, 'a') as f:
            # TODO: Add a compression protocol
            for aud in self.auds:
                print(aud, file=f)

        path = f'{campaign}/db/agents/{self.name}.pkl'
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, 'wb') as f:
            pickle.dump(self, f)

        self.auds = []

dl = Duologue(agent1_name, agent2_name)
dl.run()
