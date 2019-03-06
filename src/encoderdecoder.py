import torch
import torch.nn as nn
import torch.nn.functional as F

class EncoderModel(nn.Module):
    def __init__(self, filler_set_size, hidden_dim):
        super(EncoderModel, self).__init__()
        self.filler_set_size = filler_set_size
        self.hidden_dim = hidden_dim

        self.embedding = nn.Embedding(self.filler_set_size, self.hidden_dim)
        self.gru = nn.GRU(self.hidden_dim, self.hidden_dim)

    def forward(self, input_seq, hidden):
        embeds = self.embedding(input_seq)
        output = embeds.view(1, 1, -1)
        output, hidden = self.gru(output, hidden)
        return output, hidden


class DecoderModel(nn.Module):
    def __init__(self, hidden_dim, filler_set_size):
        super(DecoderModel, self).__init__()
        self.hidden_dim = hidden_dim

        self.embedding = nn.Embedding(filler_set_size, self.hidden_dim)
        self.gru = nn.GRU(self.hidden_dim, self.hidden_dim)
        self.out_layer = nn.Linear(self.hidden_dim, filler_set_size)
        self.softmax = nn.LogSoftmax(dim=1)

    def forward(self, input, hidden):
        output = self.embedding(input).view(1, 1, -1)
        output = F.relu(output)
        output, hidden = self.gru(output, hidden)
        output = self.softmax(self.out_layer(output[0]))
        return output, hidden

