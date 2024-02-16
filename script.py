import gensim
from sklearn.metrics.pairwise import cosine_similarity
import time
import pandas as pd
import numpy as np

model_address = '../GoogleNews-vectors-negative300.bin.gz'
dataset_address = './dataset.csv'

# Load Word2Vec model

word2vec_model = gensim.models.KeyedVectors.load_word2vec_format(
    model_address,
    binary=True,
    limit=20000 # Limit of words (GoogleNews vector contains words with their prevalence of use)
)

# 1.Preprocess the Dataset

def createDataFrame(csv_file_address):
    return pd.read_csv(csv_file_address, usecols=[1])

def preProcessDataFrame(data_frame):
    return data_frame.content.apply(gensim.utils.simple_preprocess)

data_frame = createDataFrame(dataset_address)
preprocessed_data_frame = preProcessDataFrame(data_frame)

# 2.Utilize Word Embedding Model

def get_embeddings(tokens, model):
    embeddings = [model[token] for token in tokens if token in model]
    if embeddings:
        return np.mean(embeddings, axis=0)
    else: # If can't find any match in model for any of the words.
        result = np.zeros(model.vector_size)
        result[0] = -1 # Set the first to -1 to be able to detect easily.
        return result

# 3. Retrieval System

# Use word embedding
def get_query_embedding(query, model):
    tokens = list(gensim.utils.tokenize(query, to_lower=True, deacc=True))
    embedding = get_embeddings(tokens, model)
    return embedding

# Find similarities using sklearn
def compute_similarity(query_embedding, note_embeddings):
    similarities = cosine_similarity([query_embedding], note_embeddings)
    return similarities[0]

# Find the most relevant note
def retrieve_most_relevant_note_index(query, note_embeddings, model):
    query_embedding = get_query_embedding(query, model)
    if query_embedding[0] == -1: # If can't find any match in model for any of the words
        return -1
    similarities = compute_similarity(query_embedding, note_embeddings)
    most_similar_index = np.argmax(similarities)
    return most_similar_index

# Embed all notes in the dataframe
note_embeddings = [get_embeddings(tokens, word2vec_model) for tokens in preprocessed_data_frame]

# Get the query
while True:
    query = input("\nJarvis-Mark2: Hello! What do you want to learn about the company?\n\nUser: ")

    # Retrieve the most relevant note
    most_relevant_note_index = retrieve_most_relevant_note_index(query, note_embeddings, word2vec_model)

    if most_relevant_note_index == -1:
        print("\nJarvis-Mark2: I couldn't find anything related in the database!")
    else:
        print("\nJarvis-Mark2: Sure! I found this on my database:\n")
        time.sleep(1)
        print(f"Data {most_relevant_note_index + 1}: {data_frame['content'][most_relevant_note_index]}")
        time.sleep(3)
