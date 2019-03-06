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
import random
import os

import encoderdecoder

class Trainer:
    def __init__(self, data):
        self.data = data
        filler_set_size = len(data.fillerBOW) + 1
        self.encoder = encoderdecoder.EncoderModel(filler_set_size, HIDDEN_DIM)
        self.decoder = encoderdecoder.DecoderModel(HIDDEN_DIM, filler_set_size)
        if os.path.isfile("encoderstate"):
            self.encoder.load_state_dict(torch.load("encoderstate"))
        if os.path.isfile("decoderstate"):
            self.decoder.load_state_dict(torch.load("decoderstate"))
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
            torch.save(self.encoder.state_dict(), "encoderstate")
            torch.save(self.decoder.state_dict(), "decoderstate")

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
            decoder_output, decoder_hidden = self.decoder(decoder_input, decoder_hidden)
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


