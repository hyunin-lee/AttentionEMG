import torch
import torch.nn as nn
import numpy as np
from torch.autograd import Variable

class ANN(nn.Module):
    def __init__(self, num_output, input_size, hidden_size, device):
        super(LSTM_net, self).__init__()
        self.device = device
        self.num_output = num_output
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.outlayer = nn.Linear(hidden_size, num_output)

    def forward(self, x):
        h = self.fc1(x).relu()
        h = self.fc2(h).relu()
        predict = self.outlayer(h)
        return predict


class LSTM_net(nn.Module):
    def __init__(self, num_output, input_size, hidden_size, linear_size, num_layers, seq_length, device):
        super(LSTM_net, self).__init__()
        self.device = device
        self.num_output = num_output
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.seq_length = seq_length
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size = input_size, hidden_size = hidden_size,
                            num_layers = num_layers, dropout = 0.3, bidirectional = False)
        self.fclayer = nn.Linear(hidden_size, linear_size)
        self.outlayer = nn.Linear(linear_size, num_output)

    def forward(self, x):
        scaler = 2 if self.lstm.bidirectional == True else 1
        h_state = Variable(torch.zeros(self.num_layers*scaler, x.size(0),
                                       self.hidden_size, requires_grad = True)).to(self.device)
        c_state = Variable(torch.zeros(self.num_layers*scaler, x.size(0),
                                       self.hidden_size, requires_grad = True)).to(self.device)

        lstm_out, (h, c) = self.lstm(x.transpose(1,0), (h_state, c_state))
        h = h[-1]
        h = self.fclayer(h).relu()
        predict = self.outlayer(h)
        return predict


class CNN_1D(nn.Module):
    def __init__(self, num_output, input_size, hidden_size, linear_size, kernel_size, seq_length, device):
        super(CNN_1D, self).__init__()
        self.device = device
        self.conv1 = nn.Conv1d(in_channels = input_size, out_channels = hidden_size, kernel_size = kernel_size)
        self.fclayer = nn.Linear(hidden_size * (seq_length - kernel_size + 1), linear_size)
        self.outlayer = nn.Linear(linear_size, num_output)

    def forward(self, x):
        x = self.conv1(x.transpose(1,2)).flatten(1)
        x = self.fclayer(x).relu()
        predict = self.outlayer(x)
        return predict

class seq2seq(nn.Module):
    def __init__(self, enc_size, dec_size, hidden_size, num_layers, seq_length_enc, seq_length_dec, device):
        super(seq2seq, self).__init__()
        self.device = device
        self.hidden_size = hidden_size
        self.seq_length_enc = seq_length_enc
        self.seq_length_dec = seq_length_dec
        self.num_layers = num_layers
        self.lstm_enc = nn.LSTM(input_size = enc_size, hidden_size = hidden_size,
                            num_layers = num_layers, dropout = 0.3, bidirectional = True)
        self.lstm_dec = nn.LSTM(input_size = dec_size, hidden_size = hidden_size,
                                num_layers = num_layers * (2 if self.lstm_enc.bidirectional == True else 1),
                                dropout = 0.3, bidirectional = False)
        self.fclayer1 = nn.Linear(hidden_size, hidden_size)
        self.fclayer2 = nn.Linear(hidden_size, dec_size)

    def forward(self, x, y, test = False):
        scaler = 2 if self.lstm_enc.bidirectional == True else 1
        h_state = Variable(torch.zeros(self.num_layers*scaler, x.size(0),
                                       self.hidden_size, requires_grad = True)).to(self.device)
        c_state = Variable(torch.zeros(self.num_layers*scaler, x.size(0),
                                       self.hidden_size, requires_grad = True)).to(self.device)

        _, (h, c) = self.lstm_enc(x.transpose(1,0), (h_state, c_state))

        if test:
            predict = torch.zeros(y.size(1), 1, y.size(2)).to(self.device)
            pred_in = torch.zeros(y.size(1), 1, y.size(2)).to(self.device)
            pred_in[0,0,:] = y[0, 0, :].unsqueeze(1).transpose(1, 0)

            for i in range(y.size(1)-1):
                pred, _, = self.lstm_dec(pred_in, (h, c))
                pred = self.fclayer2(self.fclayer1(pred.relu()).relu())
                #pred_in[i+1,0,:] = y[0, i+1, :].unsqueeze(1).transpose(1, 0)
                pred_in[i+1, 0, int(pred[i,0,:].argmax())] = 1
            predict = pred
        else:
            pred, (h, c) = self.lstm_dec(y.transpose(1, 0), (h, c))
            predict = self.fclayer2(self.fclayer1(pred.relu()).relu())
        return predict

