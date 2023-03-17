import torch
import numpy
import torch
import torch.nn as nn
import torch.nn.functional as F
import time

torch.set_float32_matmul_precision('high')

class MultiHeadedAttention(nn.Module):

    def __init__(self, batch_size: int, d_k: int, d_v: int, d_model: int, h: int, seq_length: int, mask=None):
        super(MultiHeadedAttention, self).__init__()

        self.d_k = d_k
        self.d_v = d_v
        self.d_model = d_model
        self.h = h
        self.seq_length = seq_length
        self.batch_size = batch_size

        assert d_model % h == 0

        # initialize the transfer weights for Q, K, V
        self.W_Q = nn.Parameter(torch.randn(1, d_model, d_k * h))
        self.W_K = nn.Parameter(torch.randn(1, d_model, d_k * h))
        self.W_V = nn.Parameter(torch.randn(1, d_model, d_v * h))
        self.W_O = nn.Parameter(torch.randn(d_v * h, d_model))

        self.mask = mask

        self.output = torch.Tensor(batch_size, seq_length, d_model)

    def forward(self, x0):
        # generate the Q, K, V matrices
        Q = torch.matmul(x0, self.W_Q)
        K = torch.matmul(x0, self.W_K)
        V = torch.matmul(x0, self.W_V)

        # reshape the Q, K, V matrices from (batch_size, seq_length, d_k * h) to (batch_size, h, seq_length, d_k)
        Q = Q.view(self.batch_size, self.seq_length, self.h, self.d_k).permute(0, 2, 1, 3)
        K = K.view(self.batch_size, self.seq_length, self.h, self.d_k).permute(0, 2, 1, 3)
        V = V.view(self.batch_size, self.seq_length, self.h, self.d_v).permute(0, 2, 1, 3)

        # calculate the attention score
        attentionScore = torch.matmul(Q, K.permute(0, 1, 3, 2)) / numpy.sqrt(self.d_k) / numpy.sqrt(self.d_v)

        # apply the mask
        if self.mask is not None:
            attentionScore = attentionScore.masked_fill(self.mask == 0, -1e9)

        # apply the softmax
        attentionScore = F.softmax(attentionScore, dim=-1)

        # calculate the output
        output = torch.matmul(attentionScore, V)
        output = output.permute(0, 2, 1, 3).contiguous().view(self.batch_size, self.seq_length, self.d_v * self.h)
        output = torch.matmul(output, self.W_O)

        return output

    def to(self, device):
        self.W_Q = nn.Parameter(self.W_Q.to(device))
        self.W_K = nn.Parameter(self.W_K.to(device))
        self.W_V = nn.Parameter(self.W_V.to(device))
        self.W_O = nn.Parameter(self.W_O.to(device))

        return self


class FeedForwardNetwork(nn.Module):

    def __init__(self, d_model: int, d_ff: int):
        super(FeedForwardNetwork, self).__init__()

        self.d_model = d_model
        self.d_ff = d_ff

        self.nn1 = nn.Linear(d_model, d_ff, bias=True)
        self.nn2 = nn.Linear(d_ff, d_model, bias=True)

    def forward(self, x0):
        output = self.nn1(x0)
        output = F.relu(output)
        output = self.nn2(output)
        return output


class EncoderBlock(nn.Module):

    def __init__(self, batch_size: int, d_k: int, d_v: int, d_model: int, h: int, seq_length: int, d_ff: int,
                 mask=None):
        super(EncoderBlock, self).__init__()

        # initialize the parameters
        self.batch_size = batch_size
        self.d_k = d_k
        self.d_v = d_v
        self.d_model = d_model
        self.h = h
        self.seq_length = seq_length
        self.d_ff = d_ff
        self.mask = mask

        # initialize the multi-headed attention
        self.multiHeadedAttention = MultiHeadedAttention(batch_size, d_k, d_v, d_model, h, seq_length, mask)

        # initialize the feed forward network
        self.feedForwardNetwork = FeedForwardNetwork(d_model, d_ff)

        # initialize the layer normalization
        self.layerNorm1 = nn.LayerNorm(d_model)
        self.layerNorm2 = nn.LayerNorm(d_model)

    def forward(self, x0):
        # calculate the multi-headed attention
        output = self.multiHeadedAttention(x0)

        # add the residual connection
        output = self.layerNorm1(output + x0)

        # calculate the feed forward network
        output = self.feedForwardNetwork(output)

        # add the residual connection
        output = self.layerNorm2(output + output)

        return output

    def to(self, device):
        self.multiHeadedAttention = self.multiHeadedAttention.to(device)
        self.feedForwardNetwork = self.feedForwardNetwork.to(device)
        self.layerNorm1 = self.layerNorm1.to(device)
        self.layerNorm2 = self.layerNorm2.to(device)

        return self


if __name__ == '__main__':
    batch_size = 10
    d_k = 64
    d_v = 64
    d_model = 512
    h = 8
    seq_length = 400

    x0 = torch.randn(batch_size, seq_length, d_model).to('cuda:0')
    model = EncoderBlock(batch_size, d_k, d_v, d_model, h, seq_length, d_ff=2048).to('cuda:0')
    # model = torch.compile(model)

    # start timer
    start = time.time()

    for _ in range(1000):
        output = model(x0)

    # end timer
    end = time.time()

    # print time in milliseconds
    print((end - start) * 1000)
    print(output.shape)
