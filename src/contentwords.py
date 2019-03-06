HIDDEN_DIM = 256
LEARNING_RATE = 0.01
N_EPOCHS = 30


import torch
import torch.nn as nn
import torch.optim as optim
import random
import os

class ContentWordsModel(nn.Module):
    def __init__(self, cw_set_size, hidden_dim=HIDDEN_DIM):
        super(ContentWordsModel, self).__init__()
        self.data = data

        self.embedding = nn.Embedding(self.cw_set_size, embedding_dim)
        self.hidden_layer = nn.Linear(embedding_dim, hidden_dim)
        self.out_layer = nn.Linear(hidden_dim, self.cw_set_size)

    def forward(self, inputs):
        embeds = self.embedding(inputs)
        h = self.hidden_layer(embeds)
        return nn.LogSoftmax(self.out_layer(h), dim=1)

