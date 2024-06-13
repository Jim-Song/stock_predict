from math import floor
import torch
from torch import nn
import torch.nn.functional as F
import numpy as np
import math


class FcSkipBlock(nn.Module):
    def __init__(self, dim_input, dim_output, dropout=0.2):
        super().__init__()
        self.fc1 = nn.Sequential(nn.Linear(dim_input, dim_output),
                                 nn.Dropout(dropout),
                                 )
        self.fc2 = nn.Sequential(nn.Linear(dim_output, dim_output),
                                 nn.Dropout(dropout),
                                 )
        self.active = nn.Tanh()

    def forward(self, x):
        x1 = self.fc1(x)
        x1 = self.active(x1)
        x2 = self.fc2(x1)
        x = x1 + self.active(x2)
        return x


class TransformerModule(nn.Module):
    def __init__(self, input_dim, nheads, nlayers):
        super().__init__()
        hidden_dim = input_dim
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim, nhead=nheads, dropout=0.1, batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer, nlayers, norm=None)
        # self.input_cat = nn.Linear(hidden_dim * max_num_p, hidden_dim*4)
        # self.hidden_cat = FcSkipBlock(hidden_dim*4, dropout=0)
        # self.output_cat = nn.Linear(hidden_dim*4, output_dim)

    def forward(self, x):
        # default batch in 0 dim, seq in 1 dim
        x = self.transformer_encoder(x)
        # bs = x.shape[0]
        # x = x.reshape(bs, -1)


        return x



# helper Module that adds positional encoding to the token embedding to introduce a notion of word order.
class PositionalEncoding(nn.Module):
    def __init__(self,
                 emb_size: int,
                 dropout: float,
                 maxlen: int = 5000):
        super(PositionalEncoding, self).__init__()
        den = torch.exp(- torch.arange(0, emb_size, 2)* math.log(10000) / emb_size)
        pos = torch.arange(0, maxlen).reshape(maxlen, 1)
        pos_embedding = torch.zeros((maxlen, emb_size))
        pos_embedding[:, 0::2] = torch.sin(pos * den)
        pos_embedding[:, 1::2] = torch.cos(pos * den)
        # pos_embedding = pos_embedding.unsqueeze(-2)

        self.dropout = nn.Dropout(dropout)
        self.register_buffer('pos_embedding', pos_embedding)

    def forward(self, input_tensor):
        # input_tensor has shape [batch, stocks, time, feature]
        batch_size = input_tensor.shape[0]
        stocks_size = input_tensor.shape[1]
        time_size = input_tensor.shape[2]
        # feature_size = input_tensor.shape[3]
        embeddings = self.pos_embedding[:time_size, :]
        embeddings = torch.stack([embeddings] * batch_size, dim=0)
        embeddings = torch.stack([embeddings] * stocks_size, dim=1)
        
        return self.dropout(torch.concat([input_tensor, embeddings], dim=3))


# helper Module to convert tensor of input indices into corresponding tensor of token embeddings
class TokenEmbedding(nn.Module):
    def __init__(self, vocab_size: int, emb_size):
        super(TokenEmbedding, self).__init__()
        self.embedding = nn.Embedding(vocab_size, emb_size)
        self.emb_size = emb_size

    def forward(self, tokens):
        return self.embedding(tokens.long()) * math.sqrt(self.emb_size)


class Model(nn.Module):
    def __init__(self, input_feature=6, time_size=35, stocks_num=20000):
        super(Model, self).__init__()
        self.hidden1 = 16
        self.emb_size_time = 8
        self.emb_size_stocks = 128
        self.stocks_num = stocks_num
        self.time_size = time_size # 考虑历史信息的长度
        self.feature_size = input_feature
        self.hidden2 = 256
        
        dropout = 0.0
        
        self.encoding_time = PositionalEncoding(
                        self.emb_size_time, dropout=dropout, maxlen=self.time_size)
        self.encode_input = FcSkipBlock(self.feature_size + self.emb_size_time, self.hidden1, 0.1)
        
        self.transformer_time = TransformerModule(
            input_dim=self.hidden1, nheads=4, nlayers=2)

        self.time_linear = FcSkipBlock(self.time_size * self.hidden1 + self.emb_size_stocks, self.hidden2, 0.1)
                
        self.transformer_stocks = TransformerModule(
            input_dim=self.hidden2, nheads=4, nlayers=2)
        self.encoding_stocks = TokenEmbedding(self.stocks_num, self.emb_size_stocks)
        
        self.output_linear = FcSkipBlock(self.hidden2, 14, 0.1)
        
        self.softplus = nn.Softplus()
        
        
    def forward(self, input_tensor, stock_indexes):
        # input_tensor [batch, stocks, time, feature]
        input_tensor = input_tensor.cuda()
        stock_indexes = stock_indexes.cuda()
        batch_size = input_tensor.shape[0]
        stocks_size = input_tensor.shape[1]
        time_size = input_tensor.shape[2]
        feature_size = input_tensor.shape[3]
        
        x11 = self.encoding_time(input_tensor)
        
        x1 = self.encode_input(x11)
        
        # 时间放在 dim 1, [batch * stocks, time, hidden1]
        x1 = torch.reshape(x1, [batch_size * stocks_size, time_size, self.hidden1])
        
        x2 = self.transformer_time(x1)
        
        # stocks放在 dim 1, [batch, stocks, time * hidden1]
        x21 = torch.reshape(x2, [batch_size, stocks_size, self.hidden1 * time_size])
        
        x22 = self.encoding_stocks(stock_indexes)
        x3 = torch.concat([x21, x22], dim=2)
                
        x4 = self.time_linear(x3)
        
        
        
        x5 = self.transformer_stocks(x4)
        x6 = torch.mean(x5, dim=1)
        
        x7 = self.output_linear(x6)
        x8 = self.softplus(x7)
        
        return x8

        

        
                
        
        
        
        
        
        
        
        
        
        
        










