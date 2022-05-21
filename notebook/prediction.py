import numpy as np
import pandas as pd
from torch import nn
import time
import os
import torch
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
    if choice == "RoBERTa":
        model.eval()
        for sent, label in testing_loader:
            sent = sent.squeeze(1)
            lst_true.append(label)
            if torch.cuda.is_available():
                sent = sent.cuda()

            with torch.no_grad():
                output = model(sent)[0]
                _, pred_label = torch.max(output.data, 1)

                lst_prediction.append(pred_label)

    else:
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

                logits = torch.argmax(outputs.logits, dim=1)
                preds = logits.detach().cpu().numpy()
                labels_ids = labels.to("cpu").numpy()

                lst_prediction.append(preds)
                lst_true.append(labels_ids)
        print("Evaluate End!")

    lst_true = [int(i) for l in lst_true for i in l]
    lst_prediction = [int(i) for l in lst_prediction for i in l]

    acc, f1_micro, f1_macro = get_result(lst_prediction, lst_true)
    cache_info(final_file, f"acc: {acc}, f1_micro: {f1_micro}, f1_macro: {f1_macro}")
