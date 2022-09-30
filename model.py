# Author: Xinwen Xu
# Date: 2022/8/1

from turtle import forward
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from dataloader import convert_to_tensor

'''
Parameters elucidation:
prefix b corresponds to 'bursts', s corresponds to strokes, which are one symbol in 'raw bursts'. 

EMBEDDING:
dict_size: total number of words/strokes occur in the corpora.
embedding_size: which also be the input size of the LSTMs in first phase of model.

FIRST PHRASE:
embedding_size: b=200, s=50
prefix_hidden_size
cat_size
inp_size: output size of 1st linear layer, also the input size of LSTM in 2nd phrase.

SECOND PHRASE:
inp_size
hidden_size
output_size=1
'''

class LSTMs(nn.Module):
    def __init__(self, b_dict_size, s_dict_size, 
                        b_hidden_size, s_hidden_size, 
                        inp_size, hidden_size,
                        device):
        super().__init__()
        self.b_emb_size = 100
        self.s_emb_size = 40

        self.id = None
        self.h = torch.zeros(1,1,hidden_size).to(device)
        self.c = torch.zeros(1,1,hidden_size).to(device)

        self.b_embedding = nn.Embedding(b_dict_size, self.b_emb_size)
        self.s_embedding = nn.Embedding(s_dict_size, self.s_emb_size)

        # Two LSTMs to encode bursts and raw_bursts separately
        self.b_lstm = nn.LSTM(self.b_emb_size, b_hidden_size, 2, dropout=0.5)
        self.s_lstm = nn.LSTM(self.s_emb_size, s_hidden_size, 2, dropout=0.5)
        cat_size = b_hidden_size + s_hidden_size
        self.linear = nn.Linear(cat_size, inp_size)

        # The third LSTM which take concatenation of burst representation and raw burst representation
        self.lstm = nn.LSTM(inp_size, hidden_size)

        # linear layer to generate prediction value
        self.fc_1 = nn.Linear(hidden_size, 1)
       # self.fc_2 = nn.Linear(400, 1)
        self.drop = nn.Dropout(p=0.3)
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, b_inp, s_inp, id):        #(L)
        #b_inp.requires_grad=s_inp.requires_grad=True

        # embedding
        b = self.b_embedding(b_inp.unsqueeze(1))    #(L,1)-->(L,1,emb_size)
        s = self.s_embedding(s_inp.unsqueeze(1))     

        _, (b_h, _) = self.b_lstm(b)
        _, (s_h, _) = self.s_lstm(s)       # h:(1,1,hid_size)

        inp_cat = torch.cat((b_h[-1].unsqueeze(0),s_h[-1].unsqueeze(0)),dim=-1)   #h[-1]: (1,1,h(b+s))
        #inp_cat = torch.cat((b_h,s_h),dim=-1)
        inp_new = self.linear(inp_cat)        #(1,1,inp_size)

        if id==self.id:
            h,c = self.h, self.c
        else:
            h,c = torch.zeros_like(self.h), torch.zeros_like(self.c)

        output, (h_o,c_o) = self.lstm(inp_new,(h,c))    #output,h: (1,1,hidden)

        pred = self.fc_1(output.squeeze(1))
        #pred = self.fc_2(pred)
        pred = self.drop(pred)
        pred = self.sigmoid(pred)
        self.id = id
        self.h, self.c = h_o.detach(), c_o.detach()

        return pred.squeeze(0)
        
    def get_representation(self, burst, strokes, device):
        _tensor = convert_to_tensor(burst, strokes, device)
        _embedding = (self.b_embedding(_tensor[0]), self.s_embedding(_tensor[1]))
        _, (b_h, _) = self.b_lstm(_embedding[0])
        _, (s_h, _) = self.s_lstm(_embedding[1])
        rep = self.linear(torch.cat((b_h[-1].unsqueeze(0),s_h[-1].unsqueeze(0)),dim=-1))
        return rep.squeeze(0)