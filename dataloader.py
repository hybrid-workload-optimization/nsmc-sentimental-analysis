import sys
sys.path.append("..")
import os, argparse, datetime, time, re, collections
from tqdm import tqdm, trange
import json
from random import random, randrange, randint, shuffle, choice
import numpy as np
import pandas as pd

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer
from keras.preprocessing.sequence import pad_sequences
from preprocessing import Preprocessor

class MovieDataset(Dataset):
    def __init__(self, tokenizer, fpath, max_len=128, use_fast=True):
        if use_fast:
            sentences, self.labels = Preprocessor(fpath, fast=True)
            input_ids = [tokenizer.encode(x) for x in sentences]
        else:
            sentences, self.labels = Preprocessor(fpath, fast=False)
            tokenized_texts = [tokenizer.tokenize(sent) for sent in sentences]
            input_ids = [tokenizer.convert_tokens_to_ids(x) for x in tokenized_texts]

        self.input_ids = pad_sequences(input_ids, maxlen=max_len, dtype='long', truncating='post', padding='post')
        self.segments = []
        self.attention_masks = attention_masks(self.input_ids)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return (self.input_ids[idx],
                self.labels[idx],
                self.attention_masks[idx])

def attention_masks(input_ids):
    attn_masks = []
    for seq in input_ids:
        seq_mask = [float(i > 0) for i in seq]
        attn_masks.append(seq_mask)

    return torch.tensor(attn_masks)


if __name__ == "__main__":
    from model import MovieClassifier
    model = MovieClassifier()
    tokenizer = AutoTokenizer.from_pretrained('bert-base-multilingual-cased', do_lower_case=False, use_fast=True)
    tr_ds = MovieDataset(tokenizer, '/home/moo/data/naver_movie/ratings.txt')
    tr_dl = DataLoader(tr_ds, shuffle=True, batch_size=64, pin_memory=True)
    input_ids, labels, attn_masks = next(iter(tr_dl))
    print(input_ids)
    print(labels)
    print(attn_masks)
