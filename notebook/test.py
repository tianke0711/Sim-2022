import json
import pandas as pd
import numpy as np
import json, re
from tqdm import tqdm_notebook
from uuid import uuid4
import tqdm

## Torch Modules
import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch.utils.data import Dataset, DataLoader

from transformers import RobertaModel, RobertaTokenizer
from transformers import RobertaForSequenceClassification, RobertaConfig


def prepare_features(seq_1, max_seq_length = 140, zero_pad = True, include_CLS_token = True, include_SEP_token = True):
    ## Tokenzine Input
    
    tokens_a = tokenizer.tokenize(seq_1)

    ## Truncate
    if len(tokens_a) > max_seq_length - 2:
        tokens_a = tokens_a[0:(max_seq_length - 2)]
    ## Initialize Tokens
    tokens = []
    if include_CLS_token:
        tokens.append(tokenizer.cls_token)
    ## Add Tokens and separators
    for token in tokens_a:
        tokens.append(token)

    if include_SEP_token:
        tokens.append(tokenizer.sep_token)
    

    input_ids = tokenizer.convert_tokens_to_ids(tokens)

    ## Input Mask 
    input_mask = [1] * len(input_ids)
    if zero_pad:
        while len(input_ids) < max_seq_length:
            input_ids.append(0)
            input_mask.append(0)
    return torch.tensor(input_ids).unsqueeze(0), input_mask
                     

class Intents(Dataset):
    def __init__(self, dataframe):
        self.len = len(dataframe)
        self.data = dataframe
        
    def __getitem__(self, index):
        utterance = self.data.text[index]
        label = self.data.label[index]
        X, _  = prepare_features(utterance)
        y = label_to_inx[self.data.label[index]]
        return X, y
    
    def __len__(self):
        return self.len


def get_result(pred, lst_true):
    from sklearn.metrics import accuracy_score, f1_score
     
    acc = accuracy_score(lst_true, pred)
    f1_micro = f1_score(lst_true, pred, average='micro')
    f1_macro = f1_score(lst_true, pred, average='macro')
    
    return acc, f1_micro, f1_macro


if __name__ == "__main__":
    label_to_inx = {'unsustainable':0,'sustainable':1}
    config = RobertaConfig.from_pretrained('roberta-base')
    config.num_labels = 2
    tokenizer = RobertaTokenizer.from_pretrained('roberta-base')
    model = RobertaForSequenceClassification(config)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    df_train = pd.read_csv('../data/train.csv')
    df_valid = pd.read_csv('../data/valid.csv')
    training_set = Intents(df_train)
    testing_set = Intents(df_valid)

    params = {'batch_size': 16,
          'shuffle': True,
          'drop_last': False,
          'num_workers': 0}
    training_loader = DataLoader(training_set, **params)
    testing_loader = DataLoader(testing_set, **params)

    loss_function = nn.CrossEntropyLoss()
    learning_rate = 1e-05
    optimizer = optim.Adam(params=model.parameters(), lr=learning_rate)

    max_epochs = 30
    model = model.train()
    for epoch in range(max_epochs):
        print("EPOCH -- {}".format(epoch))
        for i, batch in enumerate(training_loader):
            sent, label = batch[0], batch[1]
            optimizer.zero_grad()
            sent = sent.squeeze(1)
            if torch.cuda.is_available():
                sent = sent.cuda()
                label = label.cuda()
            output = model.forward(sent)[0]
            _, predicted = torch.max(output, 1)

            loss = loss_function(output, label)
            loss.backward()
            optimizer.step()
            print(f"i = {i}")
            if i % 10 == 0:
                correct = 0
                total = 0
                for sent, label in testing_loader:
                    sent = sent.squeeze(1)
                    if torch.cuda.is_available():
                        sent = sent.cuda()
                        label = label.cuda()
                    output = model.forward(sent)[0]
                    _, predicted = torch.max(output.data, 1)
                    total += label.size(0)
                    correct += (predicted.cpu() == label.cpu()).sum()
                accuracy = 100.00 * correct / total
                print('Iteration: {}. Loss: {}. Accuracy: {}%'.format(i, loss.item(), accuracy))
