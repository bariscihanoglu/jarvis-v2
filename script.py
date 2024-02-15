import gensim
from gensim.models import KeyedVectors
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd
import numpy as np

# 1.Preprocess the Dataset

def createDataFrame(csvFileAddress):
    return pd.read_csv(csvFileAddress, usecols = [1])

def preProcessDataFrame(dataFrame):
    return dataFrame.content.apply(gensim.utils.simple_preprocess)

dataFrame = createDataFrame('./dataset.csv')

preprocessedDataFrame = preProcessDataFrame(dataFrame)

# 2.Utilize Word Embedding Model

def getModel(preTrainedModelAddress):
    word2vec = gensim.models.KeyedVectors.load_word2vec_format(
        preTrainedModelAddress,
        binary=True,
        limit=1000 # to work faster
    )
    return word2vec

def get_embeddings(tokens):
    word_vectors = getModel('../GoogleNews-vectors-negative300.bin.gz')
    embeddings = []
    for token in tokens:
        if token in word_vectors:
            embeddings.append(word_vectors[token])
    if embeddings:
        return np.mean(embeddings, axis=0)  # Average the embeddings
    else:
        return np.zeros(word_vectors.vector_size)

# 3.Retrieval System

query = "office at 5am"

# Preprocess and get the embedding for the query
def get_query_embedding(query):
    tokens = list(gensim.utils.tokenize(query, to_lower=True, deacc = True))
    embedding = get_embeddings(query)
    return embedding

# Compute similarity between the query and each note
def compute_similarity(query_embedding, note_embeddings):
    similarities = cosine_similarity([query_embedding], note_embeddings)
    return similarities[0]

# Find the most relevant note
def retrieve_most_relevant_note(query, dataFrame, note_embeddings):
    query_embedding = get_query_embedding(query)
    similarities = compute_similarity(query_embedding, note_embeddings)
    most_similar_index = np.argmax(similarities)
    return most_similar_index

# Embed all notes in the dataframe
note_embeddings = [get_embeddings(tokens) for tokens in preprocessedDataFrame]

# print(dataFrame[1])

print(retrieve_most_relevant_note(
    query,
    preprocessedDataFrame,
    note_embeddings
))

# print(dataFrame[retrieve_most_relevant_note(
#     query,
#     preprocessedDataFrame,
#     note_embeddings
# )])