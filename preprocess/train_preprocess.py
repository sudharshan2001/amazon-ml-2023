import numpy as np 
import pandas as pd 
import multiprocessing as mp
import gc
from tqdm import tqdm
import spacy

df = pd.read_csv("train.csv")

df['DESCRIPTION'].fillna(' ', axis=0, inplace=True)
gc.collect()

nlp = spacy.load("en_core_web_sm")

def is_adj_noun_num(token):
    return (token.pos_ in ['ADJ', 'NOUN'] or token.like_num)

def extract_adj_noun_num(doc):
    doc = nlp(doc)
    return " ".join([token.text for token in doc if is_adj_noun_num(token)])

def extract_from_df(df):
    tqdm.pandas()
    df['NEW_DESCRIPTION'] = df['DESCRIPTION'].progress_apply(extract_adj_noun_num)
    return df

num_cores = mp.cpu_count()

chunks = [df[i:i+num_cores] for i in range(0, len(df), num_cores)]

pool = mp.Pool(num_cores)

results = []
for result in tqdm(pool.imap_unordered(extract_from_df, chunks), total=len(chunks)):
    results.append(result)

final_df = pd.concat(results)

final_df.to_csv('train_process1.csv',index=False )