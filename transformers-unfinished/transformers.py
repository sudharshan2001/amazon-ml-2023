import numpy as np 
import pandas as pd 

from transformers import AutoConfig, AutoTokenizer
from torch.utils.data import Dataset, DataLoader
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from torch.utils.data import  DataLoader
from dataset import DatasetPipeline
from bertregressor import BertRegresser
from utils import train, evaluate
from config.cfg import CFG

df = pd.read_csv("./processed files/test_preprocess1.csv")

df['NEW_DESCRIPTION'] = df['NEW_DESCRIPTION'].apply(str)

df['LEN_NEW_DESCRIPTION'] = df['NEW_DESCRIPTION'].apply(lambda x:len(x.split(' ')))

df = df[df["LEN_NEW_DESCRIPTION"]>19]
df = df[df["LEN_NEW_DESCRIPTION"]<105]

train_data, validation = train_test_split(df, test_size=0.3, random_state=21)

tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
config = AutoConfig.from_pretrained('bert-base-uncased')   
model = BertRegresser.from_pretrained('bert-base-uncased', config=config)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = model.to(device)

criterion = nn.MSELoss()

optimizer = optim.Adam(params=model.parameters(), lr=1e-3)

MAX_LEN = 120
BATCH_SIZE = 128
NUM_THREADS = 3
EPOCHS = 5

train_set = DatasetPipeline(data=train_data, maxlen=MAX_LEN, tokenizer=tokenizer)
valid_set = DatasetPipeline(data=validation, maxlen=MAX_LEN, tokenizer=tokenizer)

train_loader = DataLoader(dataset=train_set, batch_size=BATCH_SIZE, num_workers=NUM_THREADS)
valid_loader = DataLoader(dataset=valid_set, batch_size=BATCH_SIZE, num_workers=NUM_THREADS)

del df, train_data, validation 

model_final = train(model=model, 
      criterion=criterion,
      optimizer=optimizer, 
      train_loader=train_loader,
      val_loader=valid_loader,
      epochs = EPOCHS,
     device = device)

torch.save(model_final.state_dict(), './models/model.pth')