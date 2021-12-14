import torch
import torch.nn as nn
from transformers import AutoTokenizer, BertTokenizer, AutoModelForSequenceClassification
from transformers import BertForSequenceClassification, AdamW, BertConfig
from transformers import get_linear_schedule_with_warmup
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
from keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split
import sentencepiece as spm
import argparse
import pandas as pd
import numpy as np
import random
import time
import datetime

from model import MovieClassifier
from dataloader import MovieDataset
from tqdm import tqdm
from train import train
from inference import evaluate

def main(model, optimizer, epochs, tr_dl, tst_dl, scheduler):
    train(model, optimizer, epochs, tr_dl, scheduler)
    evaluate(model, tst_dl)


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--use_baseloader', action='store_true')
    parser.add_argument('--use_fastloader', action='store_true')
    
    args = parser.parse_args()

    if args.use_fastloader:
        templates= '--input={} \
        --pad_id={} \
        --bos_id={} \
        --eos_id={} \
        --unk_id={} \
        --model_prefix={} \
        --vocab_size={} \
        --character_coverage={} \
        --model_type={}'


        train_input_file = "./ratings.txt"
        vocab_size = 35000
        ### input nsmc model path into perfix
        prefix = "" 
        pad_id=0
        bos_id=1
        eos_id=2
        unk_id=3
        character_coverage = 1.0
        model_type ='word'

        cmd = templates.format(train_input_file,
                        pad_id,
                        bos_id,
                        eos_id,
                        unk_id,
                        prefix,
                        vocab_size,
                        character_coverage,
                        model_type)

        tokenizer = spm.SentencePieceProcessor(model_file=f"{prefix}.model")
        tr_ds = MovieDataset(tokenizer, './ratings.txt', use_fast=True)
        tr_dl = DataLoader(tr_ds, batch_size=128, shuffle=True, pin_memory=True)
        tst_ds = MovieDataset(tokenizer, './ratings.txt', use_fast=True)
        tst_dl = DataLoader(tst_ds, batch_size=128, shuffle=False, pin_memory=True)

    
    elif args.use_baseloader:
        tokenizer = AutoTokenizer.from_pretrained('bert-base-multilingual-cased', do_lower_case=False, use_fast=False)
        tr_ds = MovieDataset(tokenizer, './ratings.txt', use_fast=False)
        tr_dl = DataLoader(tr_ds, batch_size=128, shuffle=True, pin_memory=False)
        tst_ds = MovieDataset(tokenizer, './ratings.txt')
        tst_dl = DataLoader(tst_ds, batch_size=128, shuffle=False, pin_memory=False)
    
    model = AutoModelForSequenceClassification.from_pretrained('bert-base-multilingual-cased')
    model.cuda()
    optimizer = AdamW(model.parameters(), lr=2e-5, eps=1e-8)

    criterion = nn.CrossEntropyLoss()
    epochs = 1
    total_steps = len(tr_dl) * epochs

    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0, num_training_steps=total_steps)

    main(model, optimizer, epochs, tr_dl, tst_dl, scheduler)
