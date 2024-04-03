import gensim
from sklearn.metrics.pairwise import cosine_similarity
import time
import pandas as pd
import numpy as np

# Give necessary addressses for model and dataset.
model_address = '../GoogleNews-vectors-negative300.bin.gz'
dataset_address = './dataset.csv'

# How many output do you want to see? (Maximum)
retrieve_count = 3

# 1.Preprocess the Dataset
def createDataFrame(csv_file_address):
    return pd.read_csv(csv_file_address, usecols=[1])


def preProcessDataFrame(data_frame):
    return data_frame.content.apply(gensim.utils.simple_preprocess)


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
    if query_embedding[0] == -1:
        return []
    
    similarities = compute_similarity(query_embedding, note_embeddings)
    most_similar_indexes = np.argsort(similarities)[::-1][:retrieve_count]
    most_similar_indexes = [i for i in most_similar_indexes if similarities[i] > 0.3]
    
    # use this one to retrieve only the most relevant index.
    # most_similar_indexes = np.argmax(similarities)
    return most_similar_indexes


# EXECUTION

if __name__ == '__main__':

    # Load Word2Vec model
    word2vec_model = gensim.models.KeyedVectors.load_word2vec_format(
        model_address,
        binary=True,
        limit=20000 # Limit of words (GoogleNews vector contains words with their prevalence of use)
    )

    # Create data frame and preprocess the frame
    data_frame = createDataFrame(dataset_address)
    preprocessed_data_frame = preProcessDataFrame(data_frame)

    # Embed all notes in the dataframe
    note_embeddings = [get_embeddings(tokens, word2vec_model) for tokens in preprocessed_data_frame]

    # Get the query
    while True:
        query = input("\nJarvis-Mark2: Hello! What do you want to learn about the company?\n\nUser: ")

        # Retrieve the most relevant note
        most_relevant_note_indexes = retrieve_most_relevant_note_index(query, note_embeddings, word2vec_model)

        if not (most_relevant_note_indexes):
            time.sleep(1)
            print("\nJarvis-Mark2: I couldn't find anything related in the database!")
            time.sleep(3)
        else:
            print("\nJarvis-Mark2: Sure! I found this on my database:\n")
            time.sleep(1)
            for most_relevant_note_index in most_relevant_note_indexes:
                print(f"Data {most_relevant_note_index + 1}: {data_frame['content'][most_relevant_note_index]}")
            time.sleep(3)
