import torch
import torch.nn as nn
from transformers import AutoTokenizer, BertTokenizer, AutoModelForSequenceClassification
from transformers import BertForSequenceClassification, AdamW, BertConfig
from transformers import get_linear_schedule_with_warmup
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
from keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split

import pandas as pd
import numpy as np
import random
import time
import datetime

from model import MovieClassifier
from dataloader import MovieDataset
from metrics import accuracy
from tqdm import tqdm

def evaluate(model, tst_dl):
    model.eval()
    total_val_loss, total_val_acc = 0.0, 0.0
    for step, batch in enumerate(tqdm(tst_dl)):
        inputs, labels, attn_masks = map(lambda x: x.cuda(), batch)
        with torch.no_grad():
            outputs = model(inputs, attention_mask=attn_masks)

        logits = outputs[0]
        logits = logits.detach().cpu().numpy()
        label_ids = labels.to('cpu').numpy()
        
        eval_acc = accuracy(logits, label_ids)
        total_val_acc += eval_acc
    print(f"Accuracy: {total_val_acc / step}")


def accuracy(preds, labels):
    pred = np.argmax(preds, axis=1).flatten()
    labels = labels.flatten()

    return np.sum(pred == labels) / len(labels)
