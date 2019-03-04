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

import torch
import torch.nn as nn

class TemplateModel:
    def __init__(self, data):
        #super(RNN, self).__init__()
        self.punchline_length = data.punchline_length
        self.opening_line_length = data.opening_line_length
        self.filler_set_size = len(data.fillerBOW)

        #self.l1 = nn.Linear(self.opening_line_length)

        #self.out_layer = nn.Linear(x, )

    #def forward(self, input):
        #embeds = self.embedding(inputs)

