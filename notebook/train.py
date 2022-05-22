# Python basic
import json
import pandas as pd
import numpy as np
import time
import os

# Torch Modules
import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from torch.autograd import Variable
from torch.utils.data import Dataset, DataLoader

# transformers Modules
from transformers import RobertaModel, RobertaTokenizer
from transformers import RobertaForSequenceClassification, RobertaConfig, get_linear_schedule_with_warmup
from transformers.utils.notebook import format_time
from transformers.modeling_outputs import SequenceClassifierOutput
from transformers import AlbertTokenizer, AlbertModel, AlbertPreTrainedModel, AlbertConfig, AlbertForPreTraining
from transformers import logging

# sklearn Modules
from sklearn.model_selection import KFold

# my file
from process_file import InputDataSet, TestInput
from prediction import my_prediction

logging.set_verbosity_error()


class ALBertForSeq(AlbertPreTrainedModel):

    def __init__(self, config):
        super(ALBertForSeq, self).__init__(config)

        self.config = AlbertConfig(config)
        self.num_labels = 2
        self.albert = AlbertModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

        self.classifier = nn.Linear(config.hidden_size, self.num_labels)

        self.init_weights()

    def forward(self, input_ids, attention_mask=None, token_type_ids=None, labels=None, return_dict=None):
        """前向传播"""
        # 若return_dict不是None的话则会返回一个字典，否则返回一个字符串
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        outputs = self.albert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            return_dict=return_dict
        )

        pooler_output = outputs[1]
        pooler_output = self.dropout(pooler_output)

        logits = self.classifier(pooler_output)
        loss = None

        if labels is not None:
            loss_fnt = nn.CrossEntropyLoss()
            loss = loss_fnt(logits.view(-1, self.num_labels), labels.view(-1))

        if not return_dict:
            output = (logits,) + outputs[2:]
            return ((loss,) + output) if loss is not None else output

        return SequenceClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions
        )


def model_init(choice='RoBERTa'):
    if choice == 'RoBERTa':
        config = RobertaConfig.from_pretrained('roberta-base')
        config.num_labels = 2
        tokenizer = RobertaTokenizer.from_pretrained('roberta-base')
        model = RobertaForSequenceClassification(config)
        # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = model.cuda()
        return model, tokenizer
    else:
        model = ALBertForSeq.from_pretrained('albert-base-v2')
        tokenizer = AlbertTokenizer.from_pretrained('albert-base-v2')
        model = model.cuda()
        return model, tokenizer


def data_init(train_path, valid_path, test_path, batch_size, choice='RoBERTa'):
    if choice == 'RoBERTa':
        df_train = pd.read_csv(train_path)
        df_valid = pd.read_csv(valid_path)
        training_set = Intents(df_train)
        testing_set = Intents(df_valid)

        params = {'batch_size': batch_size,
                  'shuffle': True,
                  'drop_last': False,
                  'num_workers': 0}
        training_iter = DataLoader(training_set, **params)
        testing_iter = DataLoader(testing_set, **params)
        return training_iter, testing_iter
    else:
        train = pd.read_csv(train_path)
        val = pd.read_csv(valid_path)
        test = pd.read_csv(test_path)
        train_data = InputDataSet(train, tokenizer, 128)
        val_data = InputDataSet(val, tokenizer, 128)
        test_data = InputDataSet(test, tokenizer, 128)

        train_iter = DataLoader(train_data, batch_size=batch_size, num_workers=0)
        val_iter = DataLoader(val_data, batch_size=batch_size, num_workers=0)
        test_iter = DataLoader(test_data, batch_size=batch_size, num_workers=0)
        return train_iter, val_iter, test_iter


def cross_valid(train_path, test_path, batch_size, n_splits):
    train = pd.read_csv(train_path)
    test = pd.read_csv(test_path)
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=6)
    train_data, valid_data = [], []
    for train_idx, valid_idx in kf.split(train):
        train_temp = train.iloc[train_idx]
        valid_temp = train.iloc[valid_idx]
        train_temp.index = [x for x in range(len(train_temp))]
        valid_temp.index = [x for x in range(len(valid_temp))]
        train_data.append(train_temp)
        valid_data.append(valid_temp)
    train_iter_list = []
    for data in train_data:
        train_temp = InputDataSet(data, tokenizer, 128)
        train_iter = DataLoader(train_temp, batch_size=batch_size, num_workers=0)
        train_iter_list.append(train_iter)
    valid_iter_list = []
    for data in valid_data:
        valid_temp = InputDataSet(data, tokenizer, 128)
        valid_iter = DataLoader(valid_temp, batch_size=batch_size, num_workers=0)
        valid_iter_list.append(valid_iter)
    test_data = InputDataSet(test, tokenizer, 128)
    test_iter = DataLoader(test_data, batch_size=batch_size, num_workers=0)
    return train_iter_list, valid_iter_list, test_iter


def prepare_features(seq_1, max_seq_length=128, zero_pad=True, include_CLS_token=True, include_SEP_token=True):
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


def cache_info(out_file, text):
    """Input logging"""
    print(text)
    with open(out_file, mode="a+") as f:
        f.writelines(text + '\n')


def save_config(config, info_name):
    """Save model Super params"""
    final_file = os.path.join("../document/config", info_name + "-config.json")
    with open(final_file, mode='w') as f:
        json.dump(config, f, indent=2)
    print("Config saved!")


def train(epochs, info_name, choice):
    """Train dataSet"""

    final_file = os.path.join("../document/log", info_name + ".txt")

    start_time = time.time()

    k_result = []
    for k, (train_iter, valid_iter) in enumerate(zip(train_iter_list, valid_iter_list)):
        model, tokenizer = model_init(choice=params["choice"])
        # loss function
        loss_function = nn.CrossEntropyLoss()
        # different model use different loss func
        optimizer = optim.Adam(params=model.parameters(), lr=params["LR"]) if params["choice"] == 'RoBERTa' else AdamW(
            model.parameters(), lr=params['LR'])

        # albert need preheating model
        total_steps = len(train_iter) * epochs
        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=0.05 * total_steps,
            num_training_steps=total_steps)

        min_loss = 9999.9
        max_acc = 0
        for epoch in range(epochs):
            model.train()

            cache_info(final_file, "")
            cache_info(final_file, f"K: {k}, EPOCH -- {epoch}")
            for i, batch in enumerate(train_iter):
                if choice == "RoBERTa":
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
                else:
                    input_ids = batch["input_ids"].cuda()
                    attention_mask = batch["attention_mask"].cuda()
                    token_type_ids = batch["token_type_ids"].cuda()
                    labels = batch["labels"].cuda()

                    model.zero_grad()
                    outputs = model(input_ids, attention_mask, token_type_ids, labels)
                    loss = outputs.loss
                    loss.backward()
                    nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    optimizer.step()
                    scheduler.step()

            model.eval()
            avg_val_loss, avg_val_acc = evaluate(model, valid_iter, choice)
            cache_info(final_file, f"K: {k}, Loss: {avg_val_loss}, Acc: {avg_val_acc}")

            if min_loss > avg_val_loss or max_acc < avg_val_acc:
                min_loss = avg_val_loss
                max_acc = avg_val_acc
                output_dir = "../document/model"
                output_name = f"{info_name}-{k}-model.bin"
                output_model_file = os.path.join(output_dir, output_name)
                torch.save(model.state_dict(), output_model_file)
                print(f"Model Save!, Loss: {avg_val_loss}")

        acc, f1_micro, f1_macro = my_prediction(model, test_iter, info_name, choice=params["choice"])
        k_result.append([acc, f1_micro, f1_macro])

    avg_acc, avg_f1_mi, avg_f1_ma = 0, 0, 0
    for temp in k_result:
        avg_acc += temp[0]
        avg_f1_mi += temp[1]
        avg_f1_ma += temp[2]
    print(f"\navg_acc: {avg_acc / len(k_result)}, avg_f1_micro: {avg_f1_mi / len(k_result)}, avg_f1_macro: {avg_f1_ma / len(k_result)}")

    cache_info(final_file, f"Total train time: {format_time(time.time() - start_time)}")


def evaluate(model, val_iter, choice):
    """computer loss and acc for valid set"""
    if choice == "ALBERT":
        total_val_loss = 0
        corrects = []
        for batch in val_iter:
            # take each batch from the iterator
            input_ids = batch["input_ids"].cuda()
            attention_mask = batch["attention_mask"].cuda()
            token_type_ids = batch["token_type_ids"].cuda()
            labels = batch["labels"].cuda()

            # the outputs of the validation set are not involved in the subsequent gradient calculation of the training set
            with torch.no_grad():
                outputs = model(input_ids, attention_mask, token_type_ids, labels)

            # get the maximum value in all categories for all samples in the batch, change the maximum value to 1 and the rest to 0
            logits = torch.argmax(outputs.logits, dim=1)
            # 将预测值不参与后续训练集的梯度计算
            preds = logits.detach().cpu().numpy()
            labels_ids = labels.to("cpu").numpy()
            # get acc for now batch
            corrects.append((preds == labels_ids).mean())

            loss = outputs.loss
            # stack loss
            # total_val_loss += loss.mean().item()
            total_val_loss += loss.item()

        # get avg loss
        avg_val_loss = total_val_loss / len(val_iter)
        # get avg acc
        avg_val_acc = np.mean(corrects)
    else:
        correct = 0
        total = 0
        avg_val_loss = 0
        for sent, label in val_iter:
            sent = sent.squeeze(1)
            if torch.cuda.is_available():
                sent = sent.cuda()
                label = label.cuda()
            output = model(sent)[0]
            _, predicted = torch.max(output.data, 1)
            loss = loss_function(output, label)
            avg_val_loss += loss.item()
            total += label.size(0)
            correct += (predicted.cpu() == label.cpu()).sum()
        avg_val_loss = avg_val_loss / len(val_iter)
        avg_val_acc = correct / total

    return avg_val_loss, avg_val_acc


if __name__ == "__main__":
    # init params section
    params = {
        "batch_size": 32,
        "LR": 1e-05,
        "train_path": '../data/train_idx.csv',
        "valid_path": '../data/val_idx2.csv',
        "test_path": '../data/valid_idx.csv',
        "epochs": 6,
        "choice": 'ALBERT',
        "n_splits": 5
    }
    info_name = f"{time.strftime('%Y-%m-%d-%H-%M')}"
    label_to_inx = {'unsustainable': 0, 'sustainable': 1}

    # load model and tokenizer
    _, tokenizer = model_init(choice=params["choice"])
    # load data set
    # train_iter, valid_iter, test_iter = data_init(params["train_path"], params["valid_path"], params["test_path"], params["batch_size"], choice=params["choice"])
    train_iter_list, valid_iter_list, test_iter = cross_valid(params["train_path"], params["test_path"], params["batch_size"], params["n_splits"])

    # save config
    save_config(params, info_name)

    # train and prediction
    train(params["epochs"], info_name, choice=params["choice"])
