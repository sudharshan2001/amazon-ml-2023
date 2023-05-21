
import numpy as np 
import pandas as pd 

import os
import gensim
import multiprocessing as mp
from tqdm import tqdm
from sklearn.model_selection import train_test_split

df = pd.read_csv('./processed files/train_process1.csv')

w2v_model =  gensim.models.KeyedVectors.load('./gensim-embeddings/GoogleNews-vectors-negative300.gensim', mmap='r')

def text_to_vector(text):
    words = text.split()
    vectors = [w2v_model[w] for w in words if w in w2v_model]
    if len(vectors) > 0:
        return np.mean(vectors, axis=0)
    else:
        return np.zeros(w2v_model.vector_size)

def extract_from_df(df):
    tqdm.pandas()
    df['NEW_DESCRIPTION'] = df['DESCRIPTION'].apply(text_to_vector)
    return df


num_cores = mp.cpu_count()

chunks = [df[i:i+num_cores] for i in range(0, len(df), num_cores)]

pool = mp.Pool(num_cores)

results = []
for result in tqdm(pool.imap_unordered(extract_from_df, chunks), total=len(chunks)):
    results.append(result)

df = pd.concat(results)

df.drop(['LEN_DESCRIPTION', 'PRODUCT_TYPE_ID'], axis=1, inplace=True)

text_features_inference = []
for i in tqdm(df['NEW_DESCRIPTION'].values):
    text_features_inference.append(i)
text_features_inference = np.asarray(text_features_inference)

text_features_inference = np.concatenate([df['PRODUCT_TYPE_ID'].values.reshape(-1, 1), text_features_inference], axis=1)

np.savez_compressed('train.npz', a=text_features_inference)

df.to_csv("train.csv", index=False)