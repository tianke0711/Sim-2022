import numpy as np
import pandas as pd
from torch import nn
from torch.nn.functional import softmax
import time
import os
import torch
import json
from sklearn.metrics import classification_report


def cache_info(out_file, text):
    """Input logging"""
    print(text)
    with open(out_file, mode="a+") as f:
        f.writelines(text + '\n')


def get_result(pred, lst_true):
    """Get final result"""
    from sklearn.metrics import accuracy_score, f1_score

    acc = accuracy_score(lst_true, pred)
    f1_micro = f1_score(lst_true, pred, average='micro')
    f1_macro = f1_score(lst_true, pred, average='macro')

    return acc, f1_micro, f1_macro


def my_prediction(model, testing_loader, info_name, choice):
    """Prediction function"""

    final_file = os.path.join("../document/preds", info_name + "-preds.txt")
    lst_prediction = []
    lst_true = []
    lst_prob = []
    model = model.cuda()
    model.eval()
    print("Evaluate Start!")
    for step, batch in enumerate(testing_loader):
        print(f"The [{step + 1}]/[{len(testing_loader)}]")
        with torch.no_grad():
            input_ids = batch["input_ids"].cuda()
            attention_mask = batch["attention_mask"].cuda()
            token_type_ids = batch["token_type_ids"].cuda()
            labels = batch["labels"].cuda()

            outputs = model(input_ids, attention_mask, token_type_ids)
            probs = softmax(outputs.logits, dim=1)
            logits = torch.argmax(probs, dim=1)
            preds = logits.detach().cpu().numpy()
            labels_ids = labels.to("cpu").numpy()

            lst_prediction.append(preds)
            lst_true.append(labels_ids)
            lst_prob.append(probs)
    print("Evaluate End!")

    lst_true = [int(i) for l in lst_true for i in l]
    lst_prediction = [int(i) for l in lst_prediction for i in l]
    lst_prob = [i.to('cpu').numpy() for prob in lst_prob for i in prob]

    acc, f1_micro, f1_macro = get_result(lst_prediction, lst_true)
    cache_info(final_file, f"acc: {acc}, f1_micro: {f1_micro}, f1_macro: {f1_macro}")
    return acc, f1_micro, f1_macro, lst_prob, lst_true


def avg_prediction(k_result, lst_true, test_path):
    test = pd.read_csv(test_path)["text"].to_list()
    k_result = np.array(k_result)
    dif = []
    avg_probs = np.sum(k_result, axis=0) / 5
    avg_probs = torch.from_numpy(avg_probs)
    avg_preds = torch.argmax(avg_probs, dim=1)
    for i in range(len(lst_true)):
        if int(avg_preds[i]) != lst_true[i]:
            dif.append([test[i], lst_true[i], int(avg_preds[i])])
    df = pd.DataFrame(data=dif, columns=["text", "label", "predict"])
    df.to_csv("../data/pred_wrong.csv", index=False)
    acc, f1_micro, f1_macro = get_result(avg_preds, lst_true)
    print(f"\navg: acc: {acc}, f1_micro: {f1_micro}, f1_macro: {f1_macro}")


def real_prediction(k_result):
    k_result = np.array(k_result)
    avg_probs = np.sum(k_result, axis=0) / 5
    avg_probs = torch.from_numpy(avg_probs)
    avg_preds = torch.argmax(avg_probs, dim=1)
    df = pd.DataFrame(avg_preds, columns=['label'])
    df.to_csv('../data/real_label.csv')
    print("over!")


def getJsonFile():
    file = "../data/Sustainability_sentences_test.json"
    with open(file, mode='r') as f:
        test = f.read()
    test_data = json.loads(test)
    print(test_data[0]['sentence'])
    print(len(test_data))
    data = []
    for i in range(len(test_data)):
        sent = test_data[i]['sentence']
        data.append(sent)
    out = pd.DataFrame(data=data, columns=["text"])
    out.to_csv("../data/real_test.csv")


def save_to_Json():
    file = "../data/real_label.csv"
    label = pd.read_csv(file)['label'].to_list()
    # print(label)
    file = "../data/Sustainability_sentences_test.json"
    with open(file, mode='r') as f:
        test = f.read()
    test_data = json.loads(test)
    print(len(label))
    label_to_inx = {0: 'unsustainable', 1: 'sustainable'}
    for i in range(len(test_data)):
        test_data[i]['label'] = label_to_inx[label[i]]
    with open("../data/Sustainability_sentences_result.json", mode='w') as f:
        json.dump(test_data, f, indent=2)
    print("result saved!")


if __name__ == '__main__':
    # getJsonFile()
    save_to_Json()
