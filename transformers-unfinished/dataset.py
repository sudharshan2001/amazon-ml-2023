from transformers import BertTokenizer, AdamW
from transformers import BertPreTrainedModel, BertModel
from transformers import AutoConfig, AutoTokenizer
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

class DatasetPipeline(Dataset):

    def __init__(self, data, maxlen, tokenizer): 

        self.df = data.reset_index()
        self.tokenizer = tokenizer
        self.maxlen = maxlen

    def __len__(self):
        return self.df.shape[0]

    def __getitem__(self, index):    
        excerpt = self.df.loc[index, 'DESCRIPTION']
        try:
            target = self.df.loc[index, 'PRODUCT_LENGTH']
        except:
            target = 0.0

        tokens = self.tokenizer.tokenize(excerpt) 
        tokens = ['[CLS]'] + tokens + ['[SEP]'] 
        
        if len(tokens) < self.maxlen:
            tokens = tokens + ['[PAD]' for _ in range(self.maxlen - len(tokens))] 
        else:
            tokens = tokens[:self.maxlen-1] + ['[SEP]'] 

        input_ids = self.tokenizer.convert_tokens_to_ids(tokens) 
        input_ids = torch.tensor(input_ids) 
        attention_mask = (input_ids != 0).long()
        
        target = torch.tensor(target, dtype=torch.float32)
        
        return input_ids, attention_mask, target