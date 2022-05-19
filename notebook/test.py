import json
import pandas as pd
import numpy as np
from tqdm import tqdm_notebook
from uuid import uuid4
import tqdm
import time
import os

# Torch Modules
import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch.utils.data import Dataset, DataLoader

# transformers Modules
from transformers import RobertaModel, RobertaTokenizer
from transformers import RobertaForSequenceClassification, RobertaConfig
from transformers.utils.notebook import format_time


def model_init():
    config = RobertaConfig.from_pretrained('roberta-base')
    config.num_labels = 2
    tokenizer = RobertaTokenizer.from_pretrained('roberta-base')
    model = RobertaForSequenceClassification(config)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.cuda()
    return model, tokenizer


def data_init():
    df_train = pd.read_csv('../data/train2.csv')
    df_valid = pd.read_csv('../data/valid.csv')
    training_set = Intents(df_train)
    testing_set = Intents(df_valid)

    params = {'batch_size': 64,
              'shuffle': True,
              'drop_last': False,
              'num_workers': 0}
    training_loader = DataLoader(training_set, **params)
    testing_loader = DataLoader(testing_set, **params)
    return training_loader, testing_loader


def prepare_features(seq_1, max_seq_length=140, zero_pad=True, include_CLS_token=True, include_SEP_token=True):
    # Tokenzine Input
    tokens_a = tokenizer.tokenize(seq_1)

    # Truncate
    if len(tokens_a) > max_seq_length - 2:
        tokens_a = tokens_a[0:(max_seq_length - 2)]
    # Initialize Tokens
    tokens = []
    if include_CLS_token:
        tokens.append(tokenizer.cls_token)
    # Add Tokens and separators
    for token in tokens_a:
        tokens.append(token)

    if include_SEP_token:
        tokens.append(tokenizer.sep_token)

    input_ids = tokenizer.convert_tokens_to_ids(tokens)

    # Input Mask
    input_mask = [1] * len(input_ids)
    if zero_pad:
        while len(input_ids) < max_seq_length:
            input_ids.append(0)
            input_mask.append(0)
    return torch.tensor(input_ids).unsqueeze(0), input_mask
                     

class Intents(Dataset):
    """Get id_text and id_label"""
    def __init__(self, dataframe):
        self.len = len(dataframe)
        self.data = dataframe
        
    def __getitem__(self, index):
        utterance = self.data.text[index]
        label = self.data.label[index]
        X, _ = prepare_features(utterance)
        y = label_to_inx[self.data.label[index]]
        return X, y
    
    def __len__(self):
        return self.len


def get_result(pred, lst_true):
    """Get final result"""
    from sklearn.metrics import accuracy_score, f1_score
     
    acc = accuracy_score(lst_true, pred)
    f1_micro = f1_score(lst_true, pred, average='micro')
    f1_macro = f1_score(lst_true, pred, average='macro')
    
    return acc, f1_micro, f1_macro


def cache_info(out_file, text):
    """Input logging"""
    print(text)
    with open(out_file, mode="a+") as f:
        f.writelines(text + '\n')


def train(model, epochs, optimizer, training_loader, info_name):
    """Train dataSet"""
    max_epochs = epochs
    model = model.train()

    final_file = os.path.join("../document/log", info_name + ".txt")

    start_time = time.time()

    min_loss = 9999.9
    for epoch in range(max_epochs):
        cache_info(final_file, "")
        cache_info(final_file, f"EPOCH -- {epoch}")
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

            if min_loss > loss.item():
                min_loss = loss.item()

                output_dir = "../document/model"
                output_name = f"{info_name}-model.bin"
                output_model_file = os.path.join(output_dir, output_name)
                torch.save(model.state_dict(), output_model_file)
                print(f"Model Save!, Loss: {loss.item()}")

            if i % 100 == 0:
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
                cache_info(final_file, f"Iteration: {i}, Loss: {loss.item()}, Acc: {accuracy}")

    cache_info(final_file, f"Total train time: {format_time(time.time() - start_time)}")


def prediction(model, testing_loader, info_name):
    """Prediction function"""

    final_file = os.path.join("../document/preds", info_name + "-preds.txt")
    outputs = []
    lst_prediction = []
    lst_true = []
    lst_class = ['unsustainable', 'sustainable']
    model.eval()
    for sent, label in testing_loader:
        sent = sent.squeeze(1)
        lst_true.append(label)
        if torch.cuda.is_available():
            sent = sent.cuda()

        with torch.no_grad():
            output = model(sent)[0]
            outputs.append(output)
            _, pred_label = torch.max(output.data, 1)
            # prediction = list(label_to_inx.keys())[pred_label]
            # predicted = [lst_class[int(pred)] for pred in pred_label]
            lst_prediction.append(pred_label)
    outputs = [o.to('cpu').detach().numpy().copy() for o in outputs]

    # predictions2 = []
    # [predictions2.append([x[1] for x in [sorted(zip(example[0], lst_class), reverse=True)][0]]) for example in outputs]

    # predictions3 = [a[0] for a in predictions2]
    # lst_true = list(df_valid['label'])
    lst_true = [int(i) for l in lst_true for i in l]
    lst_prediction = [int(i) for l in lst_prediction for i in l]

    acc, f1_micro, f1_macro = get_result(lst_prediction, lst_true)
    print(f"acc: {acc}, f1_micro: {f1_micro}, f1_macro: {f1_macro}")


if __name__ == "__main__":
    label_to_inx = {'unsustainable': 0, 'sustainable': 1}

    model, tokenizer = model_init()
    training_loader, testing_loader = data_init()

    loss_function = nn.CrossEntropyLoss()
    learning_rate = 1e-05
    optimizer = optim.Adam(params=model.parameters(), lr=learning_rate)

    info_name = f"{time.strftime('%Y-%m-%d-%H-%M')}"

    train(model, 30, optimizer, training_loader, info_name)
    prediction(model, testing_loader, info_name)
