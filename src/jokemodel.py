###
# data.dataset is a list of dictionaries:
# { "openingLine": ["what", "do", "you", "call, "an", "CONTENTWORD", "CONTENTWORD","?"],
# "openingLineCWs": ["exploding", "dinosaur"],
# "t_openingLine": [130, 38, 162, 41, 51, 24376, 24376],
# "t_openingLineCWs": [16224, 20394]
# "punchline": ["a", "CONTENTWORD"],
# "punchlineCWs": ["megasaurus"],
# "t_punchline": [13, 24376],
# "t_punchlineCWs": [18565],
# "score": 56
# }
###

HIDDEN_DIM = 256
LEARNING_RATE = 0.01
N_EPOCHS = 30

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import random

class EncoderModel(nn.Module):
    def __init__(self, filler_set_size, hidden_dim):
        super(EncoderModel, self).__init__()
        self.filler_set_size = filler_set_size
        self.hidden_dim = HIDDEN_DIM

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

class Trainer:
    def __init__(self, data):
        self.data = data
        filler_set_size = len(data.fillerBOW) + 1
        self.encoder = EncoderModel(filler_set_size, HIDDEN_DIM)
        self.decoder = DecoderModel(HIDDEN_DIM, filler_set_size)
        self.learning_rate = LEARNING_RATE
        self.loss_function = nn.NLLLoss()
        self.encoder_optim = optim.SGD(self.encoder.parameters(), lr=self.learning_rate)
        self.decoder_optim = optim.SGD(self.decoder.parameters(), lr=self.learning_rate)

    def train(self, epochs=N_EPOCHS):
        for i in range(epochs):
            total_loss = 0
            dataset = self.data.dataset["train"]
            random.shuffle(dataset)

            for joke in dataset:
                input_seq = torch.tensor(joke["t_openingLine"]).view(-1, 1)
                target_seq = torch.tensor(joke["t_punchline"]).view(-1, 1)

                loss = self.trainSequence(input_seq, target_seq)
                total_loss += loss
            print("Total loss:", total_loss)
            opening_line, punchline = self.evaluate()
            print("Opening line:", opening_line)
            print("Punchline:", punchline)

    def evaluate(self):
        testset = self.data.dataset["test"]
        rand_i = random.randint(0, len(testset))
        opening_line = torch.tensor(testset[rand_i]["t_openingLine"])

        encoder_outputs = torch.zeros(self.data.opening_line_length,self.encoder.hidden_dim)
        encoder_hidden = torch.zeros(1, 1, self.encoder.hidden_dim)

        for i in range(opening_line.size(0)):
            encoder_out, encoder_hidden = self.encoder(opening_line[i], encoder_hidden)
            encoder_outputs[i] = encoder_out[0, 0]
        decoder_input = torch.tensor([[self.data.sos_token]])
        decoder_hidden = encoder_hidden
        decoded_words = []

        for i in range(self.data.punchline_length):
            decoder_output, decoder_hidden = self.decoder(decoder_input, decoder_hidden, encoder_outputs)
            topv, topi = decoder_output.data.topk(1)
            if topi.item() == self.data.eos_token:
                decoded_words.append("EMPTY")
                break
            else:
                decoded_words.append(self.data.fillerBOW[topi.item()])

            decoder_input = topi.squeeze().detach()
        return testset[rand_i]["openingLine"], " ".join(decoded_words)


    def trainSequence(self, input_seq, target_seq):
        self.encoder_optim.zero_grad()
        self.decoder_optim.zero_grad()

        encoder_outputs = torch.zeros(self.data.opening_line_length, self.encoder.hidden_dim)
        encoder_hidden = torch.zeros(1, 1, self.encoder.hidden_dim)
        loss = 0
        for i in range(input_seq.size(0)):
            encoder_out, encoder_hidden = self.encoder(input_seq[i], encoder_hidden)
            encoder_outputs = encoder_out[0, 0]

        decoder_input = torch.tensor([[self.data.sos_token]])
        decoder_hidden = encoder_hidden

        for i in range(target_seq.size(0)):
            decoder_output, decoder_hidden = self.decoder(decoder_input, decoder_hidden)
            topv, topi = decoder_output.topk(1)
            decoder_input = topi.squeeze().detach()
            loss += self.loss_function(decoder_output, target_seq[i])
            if decoder_input.item() == self.data.eos_token:
                break
        if loss == 0:
            print(input_seq, target_seq)
        loss.backward()

        self.encoder_optim.step()
        self.decoder_optim.step()

        return loss.item() / target_seq.size(0) #should probably get length of non-empty tokens


