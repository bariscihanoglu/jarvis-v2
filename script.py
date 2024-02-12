import pandas as pd
import gensim

# Preprocess the Dataset

dataframe = pd.read_csv('./dataset.csv', usecols = [1])

content = dataframe.content.apply(gensim.utils.simple_preprocess)

# Utilize Word Embedding Model

word2vec = gensim.models.KeyedVectors.load_word2vec_format('../GoogleNews-vectors-negative300.bin', binary=True)