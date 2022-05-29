import pandas as pd
import nltk
import torch
from nltk.corpus import stopwords
from transformers import AlbertTokenizer
from sklearn.model_selection import train_test_split
from random import shuffle


def remove_stopWord(file_path):
    stop_sets = set(stopwords.words('english'))
    df = pd.read_csv(file_path)
    text, label = df["text"], df["label"]
    d_text = []
    for sent in text:
        d_text.append(nltk.word_tokenize(str(sent)))
    remove_text = []
    for sent, tag in zip(d_text, label):
        temp = []
        for token in sent:
            if token not in stop_sets:
                temp.append(token)
        remove_text.append([" ".join(temp), tag])
    out = pd.DataFrame(remove_text, columns=["text", "label"])
    out.to_csv("../data/test_idx_stopword.csv", index=False)


def remove_punc(file_path):
    punc_sets = [',', '.', '!', '+', '-', '/', '*', '(', ')']
    df = pd.read_csv(file_path)
    text, label = df["text"], df["label"]
    d_text = []
    for sent in text:
        d_text.append(nltk.word_tokenize(str(sent)))
    remove_text = []
    for sent, tag in zip(d_text, label):
        temp = []
        for token in sent:
            if token not in punc_sets:
                temp.append(token)
        remove_text.append([" ".join(temp), tag])
    out = pd.DataFrame(remove_text, columns=["text", "label"])
    out.to_csv("../data/test_stopword_punc.csv", index=False)


class InputDataSet:

    def __init__(self, data, tokenizer, max_len):
        self.data = data
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        text = str(self.data['text'][index])
        labels = torch.tensor(self.data['label'][index], dtype=torch.long)

        output = self.tokenizer.encode_plus(
            text=text,
            add_special_tokens=True,
            truncation=True,
            padding='max_length',
            max_length=self.max_len,
            return_tensors='pt'
        )

        input_ids, token_type_ids, attention_mask = output.values()
        input_ids = input_ids.squeeze(dim=0)
        attention_mask = attention_mask.squeeze(dim=0)
        token_type_ids = token_type_ids.squeeze(dim=0)
        return {
            'text': text,
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'token_type_ids': token_type_ids,
            'labels': labels
        }


class TestInput:

    def __init__(self, data, tokenizer, max_len):
        self.data = data
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        text = str(self.data['text'][index])

        output = self.tokenizer.encode_plus(
            text=text,
            add_special_tokens=True,
            truncation=True,
            padding='max_length',
            max_length=self.max_len,
            return_tensors='pt'
        )

        input_ids, token_type_ids, attention_mask = output.values()
        input_ids = input_ids.squeeze(dim=0)
        attention_mask = attention_mask.squeeze(dim=0)
        token_type_ids = token_type_ids.squeeze(dim=0)
        return {
            'text': text,
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'token_type_ids': token_type_ids,
        }


def label_to_idx(file_path):
    df = pd.read_csv(file_path)
    text, label = df["text"], df["label"]

    label_to_inx = {'unsustainable': 0, 'sustainable': 1}

    out = []
    for l, t in zip(label, text):
        out.append([t, label_to_inx[l]])

    df2 = pd.DataFrame(out, columns=["text", "label"])
    df2.to_csv("../data/test_idx.csv", index=False)


def split_train_val(file_path, rate):
    df = pd.read_csv(file_path)
    labels, texts = df["label"], df["text"]

    train_data, val_data, train_label, val_label = train_test_split(texts, labels, shuffle=True, test_size=rate,
                                                                    random_state=6)

    train = zip(train_data, train_label)
    val = zip(val_data, val_label)

    train_df = pd.DataFrame(train, columns=["text", "label"])
    val_df = pd.DataFrame(val, columns=["text", "label"])

    train_df.to_csv("../data/train_stopword.csv")
    val_df.to_csv("../data/test_stopword.csv")


def merge_train_test(train_path, test_path):
    train = pd.read_csv(train_path)
    test = pd.read_csv(test_path)
    total = train.append(test)
    total.index = [x for x in range(len(total))]
    total.to_csv("../data/total_idx.csv")


if __name__ == '__main__':
    remove_stopWord("../data/test_idx.csv")
    # remove_punc("../data/test_stopword.csv")
    # label_to_idx("../data/test.csv")
    # split_train_val("../data/total_stopword.csv", 0.1)
    # merge_train_test("../data/train_idx.csv", "../data/test_idx.csv")
