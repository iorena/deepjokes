HIDDEN_DIM = 256
LEARNING_RATE = 0.01
N_EPOCHS = 30
EMBEDDING_DIM = 16


import torch
import torch.nn as nn
import torch.optim as optim
import random
import os

class ContentWordsModel(nn.Module):
    def __init__(self, set_size, cw_set_size, hidden_dim=HIDDEN_DIM):
        super(ContentWordsModel, self).__init__()

        self.hidden_dim = hidden_dim
        self.embedding = nn.Embedding(set_size, EMBEDDING_DIM)
        self.lstm = nn.LSTM(EMBEDDING_DIM, hidden_dim)
        self.hidden_layer = nn.Linear(hidden_dim, cw_set_size)
        self.softmax = nn.LogSoftmax(dim=1)

        self.hidden = self.init_hidden()

    def forward(self, inputs):
        embeds = self.embedding(inputs.view(-1))
        lstm_out, self.hidden = self.lstm(embeds.view(len(inputs), 1, -1), self.hidden)
        hidden_out = self.hidden_layer(lstm_out[-1])
        return self.softmax(hidden_out)

    def init_hidden(self):
        return (torch.zeros(1, 1, self.hidden_dim), torch.zeros(1, 1, self.hidden_dim))

class ContentWordsTrainer():
    def __init__(self, data):
        self.model = ContentWordsModel(len(data.fillerBOW) + len(data.contentBOW), len(data.contentBOW))
        self.data = data
        self.loss_function = nn.NLLLoss()
        self.optim = optim.SGD(self.model.parameters(), lr=LEARNING_RATE)
        if os.path.isfile("lstmstate"):
            self.model.load_state_dict(torch.load("lstmstate"))


    def train(self):
        for i in range(N_EPOCHS):
            total_loss = 0
            dataset = self.data.dataset["train"]
            random.shuffle(dataset)

            for joke in dataset:
                input_seq = torch.tensor(joke["t_openingLine"]).view(-1, 1)
                input_seq = torch.cat((input_seq, torch.tensor(joke["t_openingLineCWs"]).view(-1, 1)), 0)
                for token in joke["t_punchlineCWs"]:
                    target = torch.tensor(token).view(-1)
                    loss, predicted = self.trainSequence(input_seq, target)
                    input_seq = torch.cat((input_seq, predicted.view(-1, 1)), 0)
                    total_loss += loss

            print("Total loss:", total_loss.item())
            opening_line, punchline = self.evaluate()
            print("Opening line:", " ".join(opening_line))
            print("Punchline:", " ".join(punchline))
            torch.save(self.model.state_dict(), "lstmstate")


    def trainSequence(self, input_seq, target):
        self.optim.zero_grad()
        self.model.hidden = self.model.init_hidden()

        probs = self.model(input_seq)
        loss = self.loss_function(probs, target)
        _, predicted = torch.max(probs, 1)
        return loss, predicted


    def evaluate(self):
        testset = self.data.dataset["test"]
        rand_i = random.randint(0, len(testset))
        opening_line = torch.tensor(testset[rand_i]["t_openingLine"])
        opening_cws = torch.tensor(testset[rand_i]["t_openingLineCWs"])
        input_seq = torch.cat((opening_line, opening_cws), 0).view(-1, 1)
        predicted_punchline = testset[rand_i]["t_punchline"]
        predictedCWs = []
        for i in range(len(testset[rand_i]["t_punchlineCWs"])):
            probs = self.model(input_seq)
            _, predicted = torch.max(probs, 1)
            input_seq = torch.cat((input_seq, predicted.view(-1, 1)), 0)
            predictedCWs.append(predicted.item())
        i = 0
        sentence = []
        for index, token in enumerate(predicted_punchline):
            if token == self.data.cw_token:
                contentword = self.data.contentBOW[predictedCWs[i] - 1 - len(self.data.fillerBOW)]
                sentence.append(contentword)
                i += 1
            else:
                sentence.append(self.data.fillerBOW[token])
        return testset[rand_i]["openingLine"], sentence

    def predict(self, opening, punchline):
        punchline_cws = 0
        for token in punchline:
            if token == self.data.cw_token:
                punchline_cws += 1
        input_seq = torch.tensor(opening).view(-1, 1)
        predictedCWs = []
        for i in range(punchline_cws):
            probs = self.model(input_seq)
            _, predicted = torch.max(probs, 1)
            input_seq = torch.cat((input_seq, predicted.view(-1, 1)), 0)
            predictedCWs.append(predicted.item())
        predicted_punchline = []
        i = 0
        for index, token in enumerate(punchline):
            if token == self.data.cw_token:
                predicted_punchline.append(self.data.contentBOW[predictedCWs[i] - 1 - len(self.data.fillerBOW)])
                i += 1
            else:
                predicted_punchline.append(self.data.fillerBOW[token])
        opening_formatted = []
        split = opening.index(self.data.eos_token)
        opening_template = opening[:split]
        opening_cws = opening[split+1:]
        i = 0
        for token in opening_template:
            if token == self.data.cw_token:
                opening_formatted.append(opening_cws[i])
                i += 1
            else:
                opening_formatted.append(token)
        return " ".join([self.data.allBOW[token] for token in opening_formatted]), " ".join(predicted_punchline)

