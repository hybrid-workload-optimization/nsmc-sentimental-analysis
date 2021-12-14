import torch
import torch.nn as nn
from transformers import AutoTokenizer
from transformers import AutoModel, AdamW, BertConfig
from transformers import get_linear_schedule_with_warmup
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler

import pandas as pd
import numpy as np

class MovieClassifier(nn.Module):
    def __init__(self, hidden_dim=768, n_labels=5):
        super().__init__()
        self.emb = AutoModel.from_pretrained('bert-base-multilingual-cased')
        self.fc = nn.Linear(hidden_dim, n_labels)

    def forward(self, inputs):
        import pdb; pdb.set_trace()
        outputs = self.emb(inputs)
        logits = self.fc(outputs[0])

        return logits

if __name__ == "__main__":
    model = MovieClassifier()
    print(model)
