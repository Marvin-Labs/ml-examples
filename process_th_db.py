"""
This script demonstrates the handling of a thematic database as created by create_th_db.py.

The json dataset is first parsed so that we have a list of documents each mapped to an event which is a dictionary from thematic roles to their content, and from 'predicate' to the event's predicate (usually the main verb).

We then collect (i) all the possible contents of the most important thematic roles (we call them NPs, or noun phrases), because this is what they usually are, and (ii) all predicates, and embed all of them using some embedding model. In this example we are using finbert, because we are using financial data for the example, but you can use any embedding model which suits your data.




"""



from collections import defaultdict
import torch
from tqdm import tqdm
from transformers import pipeline
import json
import numpy as np
import pandas as pd

flatten = lambda l: [item for sublist in l for item in sublist]

def change_outliers_to_inds(labels):
    """Changes all the -1s to their own cluster, indexed by order."""
    n_labels = len(set(labels)) - 1
    index = n_labels + 1
    for i in range(len(labels)):
        if labels[i] == -1:
            labels[i] = index
            index += 1
    return labels

def extract_relevant_sentence(doc, event_dict):
    """
    Give a document and an event dictionary,
    return the sentence in the document which is most likely to be the one represented by the event dict.
    """
    doc = doc.replace('U.S.', 'US')
    # Tokenize the paragraph into sentences
    sentences = doc.split('. ')

    # Extract key terms from the event dictionary
    key_terms = set()
    for value in event_dict.values():
        key_terms.update(value.split())

    # Score each sentence based on the number of key terms it contains
    sentence_scores = []
    for sentence in sentences:
        score = sum(word in sentence for word in key_terms)
        sentence_scores.append((score, sentence))

    # Select the sentence with the highest score
    most_relevant_sentence = max(sentence_scores, key=lambda x: x[0])[1]
    return most_relevant_sentence

def calculate_probabilities(vectors):
    """
    This means probabilities are calculated over the probability of clusters in their thematic roles.
    One part of the probability is the position probability, that is, how likely is cluster i to appear in role TH.
    Another part is pair coocurrence, that is, how likely is cluster i to appear in role TH1 given that cluster j appears in role TH2.
    You can play around with it in many ways: take only single role probabilities, or only double role coocurrence, or consider triple role coocurrence, or experiment with different weighting methods,etc'.
    """

    # Initialize dictionaries
    position_freq = defaultdict(lambda: defaultdict(int))
    pair_freq = defaultdict(lambda: defaultdict(int))
    total_vectors = len(vectors)

    # Calculate frequency of each value at each position and pair frequency
    for vector in vectors:
        for i, value in enumerate(vector):
            position_freq[i][value] += 1
            for j in range(i+1, len(vector)):
                pair_freq[(i, j)][(value, vector[j])] += 1

    # Calculate probabilities
    position_prob = {pos: {val: count/total_vectors for val, count in values.items()}
                     for pos, values in position_freq.items()}
    pair_prob = {(pos1, pos2): {val_pair: count/total_vectors for val_pair, count in pairs.items()}
                 for (pos1, pos2), pairs in pair_freq.items()}

    # Marginalize None by setting its probability to 1
    for pos in position_prob.keys():
        position_prob[pos][None]=1

    # Assign probabilities to each vector by multiplying the probability of all of its components
    vector_probabilities = []
    for vector in vectors:
        prob = 1
        for i, value in enumerate(vector):
            prob *= position_prob[i].get(value, 0)
            for j in range(i+1, len(vector)):
                pair = (value, vector[j])
                prob *= pair_prob[(i, j)].get(pair, 0)
        vector_probabilities.append(prob)

    return position_prob, pair_prob, vector_probabilities


def find_top_k_extremes_indices(vector_probs, k):
    """
    Find k extreme indices to either side of vector probabilities returned by calculate_probabilities,
    by observing their assigned probabilities (index 1 of each pair).
    """
    # Enumerate and sort based on probabilities
    indexed_vector_probs = sorted(list(enumerate(vector_probs)), key=lambda x: x[1])

    # Get indices of top k with lowest and highest probabilities
    lowest_k_indices = [x[0] for x in indexed_vector_probs[:k]]
    highest_k_indices = [x[0] for x in indexed_vector_probs[-k:]]

    return lowest_k_indices, highest_k_indices

def embs_from_docs_pipeline(docs,embedding_model):
    """
    Use embedding_model to embed each document in docs.
    Picks the last vector in the model's output, because this is the top layer.
    If the document is too long, iteratively truncate it by a 100 characters until it fits.
    Return a matrix of all document embedding vectors.
    """
    print('Creating document embeddings...')
    embs = []
    truncs = []
    for doc in tqdm(docs):
        trunc = 3000 # Initial truncation
        while True:
            try:
                # Truncate document before entering the model, and pick the last output layer ([0,-1])
                embs.append(embedding_model(doc[:trunc],return_tensors='pt')[0, -1])
                truncs.append(trunc)
                break
            except RuntimeError:
                # If it doesn't fit, truncate by a little more
                trunc-=100
    return torch.stack(embs)

def embed_finbert(docs):
    """Use the embedding function with the finbert model."""
    pl = pipeline('feature-extraction', 'yiyanghkust/finbert-tone', device='cuda:0')
    return embs_from_docs_pipeline(docs,pl)

embed = embed_finbert # replace with your favorite embedding function (it should take an iterable of strings and return a matrix)

# This is tangential to our overall point in this example,
# but we need to filter out legal statements so we've built a classifier that does that.
from pickle import load
with open('output/is_business_rf_classifier.pkl','rb') as f:
    is_business_classifier = load(f)

# Load any other saved database that was generated using the thematic prompt
ID = '64d11d67b8a4a9b9e363ceac'
dataset_name = f'mng-rep-{ID}'
source2db = json.load(open(f'output/{dataset_name}-them.json','r'))

docs = list(source2db.keys())
db = list(source2db.values())

N = len(docs)

embs = embed(docs)
business_labels = is_business_classifier.predict(embs)

docs = [docs[i] for i in range(N) if business_labels[i]]
db = [db[i] for i in range(N) if  business_labels[i]]

# First evaluate all entries in the database from json to python
evaled_db = []
for i, doc_rep in enumerate(db):
    try:
        doc_rep = eval(doc_rep)
    except:
        doc_rep = []
    if type(doc_rep) == dict:
        doc_rep = [doc_rep]

    evaled_db.append(doc_rep)

# Collect metadata from each role to all of its possible values
# This dict maps each role in the database to a list of its values (content)
# alongside indices (i,j) representing where this role had this value within the db.
role2cont_and_pos = defaultdict(list)
for i,doc_rep in enumerate(evaled_db):
    for j,event in enumerate(doc_rep):
        for role,content in event.items():
            # collapse duplicate roles (like "Agent2", the second agent in the event)
            if role[-1] in '23456789':
                role = role[:-1]
            role2cont_and_pos[role].append((content,(i,j)))

# To get all contents for a certain role
get_conts = lambda l: [x[0] for x in l]


# Get the content of all important NPs and embed them
NPs = (get_conts(role2cont_and_pos['Agent'])\
      + get_conts(role2cont_and_pos['Patient'])\
      + get_conts(role2cont_and_pos['Goal'])\
      + get_conts(role2cont_and_pos['Recipient'])\
      + + get_conts(role2cont_and_pos['Instrument']))

NPs = list(set([x for x in NPs if type(x) is str]))
nembs = embed(NPs)

# Same for predicates
preds =  get_conts(role2cont_and_pos['Predicate'])
pred_types = list(set(preds))
pembs = embed(pred_types)


# Now we run clustering with several hyparam choices to compare them and choose the best one
# KMeans is one of the simplest ways to cluster for demonstration purposes,
# but the optimal cllustering scheme changes massively depending on the application.
from sklearn.cluster import KMeans

# NP labels
np_labels = KMeans(n_clusters=40).fit_predict(nembs)

# Map each NP to its clusters (both as a dict and a dataframe)
NP2clust = {np:label for np,label in zip(NPs, np_labels)}
something = {i:[np,label] for i,(np,label) in enumerate(zip(NPs, np_labels))}
np_clust_df = pd.DataFrame.from_dict(something, orient='index',
                                     columns=['np','cluster'])

# For each NP cluster, pick the shortest one in the cluster as a stand-in
def get_shortest(lst):
    return min(lst, key=lambda x: len(x))
clust2np = {label:get_shortest(np_clust_df[np_clust_df.cluster==label].values[:,0]) for label in np_labels}


# Display examples from each cluster
def print_cluster_exs(c, s=10):
    cluster_inds = np.where(np_clust_df.cluster == c)[0]
    sampl = np.random.choice(cluster_inds, s)
    for i in sampl:
        print(NPs[i])

for i in range(len(set(np_labels)) - 1):
    print(i)
    print_cluster_exs(i)
    print('-----------------')

# Cluster the predicates
pred_labels = KMeans(n_clusters=int(len(pembs)/4)).fit_predict(pembs)

# Create a dictionary and a dataframe mapping for predicates as well
pred_labels = change_outliers_to_inds(pred_labels)
pred2clust = {pred:label for pred,label in zip(pred_types,pred_labels)}
something = {i:[pred,label] for i,(pred,label) in enumerate(zip(pred_types,pred_labels))}
pred_clust_df = pd.DataFrame.from_dict(something, orient='index',columns=['pred','cluster'])
clust2pred = {label:pred_clust_df[pred_clust_df.cluster==label].iloc[0].pred for label in pred_labels}

# -1 stands for "unclustered" in many clustering packages
# we just convert it to a string
clust2np[-1] = '-1'
clust2pred[-1] = '-1'

# We define it here because it uses the clusters we just found
def clustered_event_dict(event):
    """Based on the found clusters, this function returns a clustered representation
    of an event to a readable one, where each cluster index is replaced by
    a representative string (shortest member of the cluster)."""
    for role, label in event.items():
        if role == 'Predicate':
            event[role] = clust2pred.get(event[role], None)
        else:
            event[role] = clust2np.get(event[role], None)
    return event


# Collapse each NP into its cluster ID to compress the dataset
collapsed_db = []
non_collapsed_db = []
for entry in evaled_db:
    new_events = []
    for event in entry:
        new_event = {}

        # Iterate over the events
        for  role, content in event.items():
            # Change every pred or NP to their cluster ID
            if role == 'Predicate':
                new_event[role] = pred2clust.get(content,-1)
            else:
                # The reason we have some cases here is because ChatGPT is not 100% consistent in its output
                # so I had to just manually account for all cases.
                # If you run into errors in this code it is likely because there of further inconsistencies in the output,
                # and unfortunately there isn't an elegant solution. But this should work in most cases:

                ignore = False
                if type(content) is dict:
                    content = list(content.keys())[0]
                    new_event[role] = NP2clust.get(content, -1)
                if type(content) is list:
                    if all([type(x) is str for x in content]):
                        content = ' and '.join(content)
                    elif all([type(x) is dict for x in content]):
                        # this case is negligible, ignore
                        ignore = True
                if not ignore:
                    new_event[role] = NP2clust.get(content, -1)
        new_events.append(new_event)
    collapsed_db.append(new_events)

# Collect metadata mapping mapping event indices to the index of their origin document
# We will use it to extract the correct document for each event
ind2doc = {}
ind=0
for i,events in enumerate(collapsed_db):
    for _ in events:
        ind2doc[ind]=i
        ind+=1

# This is a flat list of all events, not dividided by docs
events = flatten(collapsed_db)

# This is just for reference as we want to display the uncollapsed events for comprehensibility
raw_db = flatten(evaled_db)

# Small event represetnation, just as a tuple of clusters.
# You can use this as sentence embeddings, for various purpose as we do below.
small_events = [(event.get('Predicate',None),event.get('Agent',None),event.get('Patient',None),event.get('Goal',None),event.get('Recipient',None)) for event in events]

# Example usage:
# Calculate coocurrence probabilities based on the small event representations as vectors
# See function docstring for explanation
vectors = small_events
position_prob, pair_prob, vector_probs = calculate_probabilities(vectors)

# Find events with extreme k probabilities
k = 10
lowest_k_indices, highest_k_indices = find_top_k_extremes_indices(vector_probs, k)

# Display events with low probabilities, which identifies surprising/interesting events in our data
# Print the full doc from which the event is taken followed by the reconstructed event.
# The way the prompt is setup now, ChatGPT doesn't tell us which event corresponds to which sentence,
# so you have to figure it out from the event representation. But this should be pretty straightforward.
for i in lowest_k_indices:

    print('Document':)
    print(docs[ind2doc[i]])
    print()
    print('Sentence:')
    relevant_sentence = extract_relevant_sentence(paragraph, event_dict)
    print()
    print('Event:')
    print(raw_db[i])
    print()
    print('Event with clusters:')
    print(clustered_event_dict(events[i]))
    print('------------------')


