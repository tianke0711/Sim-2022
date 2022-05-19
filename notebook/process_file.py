import pandas as pd
import nltk
from nltk.corpus import stopwords
from sklearn.model_selection import train_test_split


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
    out.to_csv("../data/train2.csv", index=False)


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
    out.to_csv("../data/train4.csv", index=False)


if __name__ == '__main__':
    # remove_stopWord("../data/train.csv")
    remove_punc("../data/train2.csv")
