import pandas as pd

def Preprocessor(data, fast=True):
    if fast:
        dataset = pd.read_csv(data, sep="\t")
        sentences = dataset['document']
        labels = dataset['label'].values
        for i in range(1): sentences = ["CLS " + str(sentence) + " [SEP]" for sentence in sentences]
    else:
        dataset = pd.read_csv(data, sep="\t")
        sentences = dataset['document']
        labels = dataset['label'].values
        for i in range(1): sentences = ["CLS " + str(sentence) + " [SEP]" for sentence in sentences]
    return sentences, labels