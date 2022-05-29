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
from transformers import RobertaModel, RobertaTokenizer, RobertaPreTrainedModel
from transformers import RobertaForSequenceClassification, RobertaConfig, get_linear_schedule_with_warmup
from transformers.utils.notebook import format_time
from transformers.modeling_outputs import SequenceClassifierOutput
from transformers import AlbertTokenizer, AlbertModel, AlbertPreTrainedModel, AlbertConfig, AlbertForPreTraining
from transformers import logging

# sklearn Modules
from sklearn.model_selection import KFold

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
        # pooler_output = pooler_output[:, 0]
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


class RoBERTaForSeq(RobertaPreTrainedModel):
    def __init__(self, config):
        super(RoBERTaForSeq, self).__init__(config)

        self.config = RobertaConfig(config)
        self.num_labels = 2
        self.roberta = RobertaModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

        self.classifier = nn.Linear(config.hidden_size, self.num_labels)
        self.relu = nn.ReLU()

        self.init_weights()

    def forward(self, input_ids, attention_mask=None, token_type_ids=None, labels=None, return_dict=None):
        """前向传播"""
        # 若return_dict不是None的话则会返回一个字典，否则返回一个字符串
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        outputs = self.roberta(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            return_dict=return_dict
        )

        pooler_output = outputs[0]
        pooler_output = pooler_output[:, 0]
        pooler_output = self.dropout(pooler_output)

        logits = self.classifier(pooler_output)
        logits = self.relu(logits)
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
