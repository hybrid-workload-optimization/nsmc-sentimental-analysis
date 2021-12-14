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


def train(model, optimizer, epochs, tr_dl, scheduler):
    model.zero_grad()
    for i in tqdm(range(0, epochs)):
        total_loss = 0.0
        model.train()
        for step, batch in enumerate(tqdm(tr_dl)):

            inputs, labels, attn_masks = map(lambda x: x.cuda(), batch)
            outputs = model(inputs, attention_mask=attn_masks, labels=labels)
            loss = outputs[0]

            total_loss += loss.item()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)

            optimizer.step()
            scheduler.step()
            model.zero_grad()

        avg_tr_loss = total_loss / len(tr_dl)
        print(avg_tr_loss)
