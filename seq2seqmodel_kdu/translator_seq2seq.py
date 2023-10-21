import numpy as np
import pandas as pd
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.utils import to_categorical
import time

def load_translator_data():
    data = pd.read_csv('fra.txt', names = ['src', 'tar', 'lic'], sep='\t')
    del data['lic']
    data = data.loc[:, 'src':'tar']
    data = data[10000:80000]
    data.tar = data.tar.apply(lambda x: '\t'+x+'\n')
    print(data.head())

    src_vocab = set()
    for word in data.src:
        for char in word:
            src_vocab.add(char)

    tar_vocab = set()
    for word in data.tar:
        for char in word:
            tar_vocab.add(char)

    src_vocab = sorted(list(src_vocab))
    tar_vocab = sorted(list(tar_vocab))

    src2idx = dict([(word, i+1) for i, word in enumerate(src_vocab)])
    tar2idx = dict([(word, i+1) for i, word in enumerate(tar_vocab)])

    enc_input = []
    for datum in data.src:
        imsi = []
        for word in datum:
            imsi.append(src2idx[word])
        enc_input.append(imsi)
    print(enc_input[:3])

    dec_input = []
    for datum in data.tar:
        imsi = []
        for word in datum:
            imsi.append(tar2idx[word])
        dec_input.append(imsi)
    print(dec_input[:3])

    dec_target = []
    for datum in data.tar:
        t = 0
        imsi = []
        for word in datum:
            if t>0:
                imsi.append(tar2idx[word])
            t  = t+1
        dec_target.append(imsi)
    print(dec_target[:3])


    max_len_src = max([len(data) for data in data.src])
    max_len_tar = max([len(data) for data in data.tar])
    print(max_len_src, max_len_tar) # sequence_length
    print(len(src_vocab)+1, len(tar_vocab)+1) # feature_length

    enc_input = to_categorical(pad_sequences(enc_input, maxlen = max_len_src, padding = 'post'))
    dec_input = to_categorical(pad_sequences(dec_input, maxlen = max_len_tar, padding = 'post'))
    dec_target = pad_sequences(dec_target, maxlen = max_len_tar, padding ='post')
    print('--shapes of the dataset--')
    print(np.shape(enc_input))
    print(np.shape(dec_input))
    print(np.shape(dec_target))

    return enc_input, dec_input, dec_target, tar2idx



import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import argparse
from torch.utils.data import DataLoader, TensorDataset
from torch import LongTensor as LT
from torch import FloatTensor as FT
from model import LSTM_net, CNN_1D, seq2seq
from tqdm import tqdm
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt

parser = argparse.ArgumentParser()
parser.add_argument('--lr', type = float, default = 0.02)
parser.add_argument('--nepoch', type = int, default = 10)
parser.add_argument('--seqlen', type = int, default = 5)
args = parser.parse_args()

class Generate_Dataset(torch.utils.data.Dataset):
    def __init__(self, data_enc, data_dec, data_tar):
        self.x1_data = data_enc
        self.x2_data = data_dec
        self.y_data = data_tar
        self.device = device
    def __len__(self):
        return len(self.x1_data)
    def __getitem__(self, idx):
        x1 = FT(self.x1_data[idx]).to(self.device)
        x2 = FT(self.x2_data[idx]).to(self.device)
        y = FT(self.y_data[idx]).to(self.device)
        return x1, x2, y


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
lstm_net = seq2seq(enc_size = 79, dec_size = 101, hidden_size = 64,
                   num_layers = 1, seq_length_enc = 22, seq_length_dec = 1, device = device).to(device)

data_enc, data_dec, data_tar, tar2idx = load_translator_data()

dataset = Generate_Dataset(data_enc[:-1000], data_dec[:-1000], data_tar[:-1000])
#dataset_test = dataset
dataset_test = Generate_Dataset(data_enc[-1000:], data_dec[-1000:], data_tar[-1000:])

train_loader = DataLoader(dataset, batch_size = 512, shuffle = True)
test_loader = DataLoader(dataset_test, batch_size = 1, shuffle = False)
optimizer = torch.optim.Adam(lstm_net.parameters(), lr = args.lr)


for epoch in range(args.nepoch):
    with tqdm(train_loader, unit = 'batch') as tepoch:
        for x1, x2, y in tepoch:
            predict = lstm_net(x1, x2)
            loss = 0
            result = 0
            for i in range(74):
                loss += torch.nn.functional.cross_entropy(predict[i,:,:], y[:,i].long()) / 74
                result += torch.eq(predict[i,:,:].argmax(1), y[:,i].long()).sum().item() / 74 / 512
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            tepoch.set_description(f"Epoch {epoch}")
            tepoch.set_postfix(loss = loss.item(), acc = 100. * result)

print('Testing')
ypred = np.zeros(len(dataset_test))
ytruth = np.zeros(len(dataset_test))
index_to_tar = dict((i, char) for char, i in tar2idx.items())
with tqdm(test_loader, unit = 'batch') as tepoch:
    for i, (x1, x2, y) in enumerate(tepoch):
        translate = []
        answer = []
        predict = lstm_net(x1, x2, test=True)
        for j in range(73):
            try:
                answer_char = index_to_tar[int(y[0,j].cpu().detach().numpy())]
            except:
                break
            try:
                sampled_char = index_to_tar[int(np.argmax(predict[j,0,:].cpu().detach().numpy()))]
            except:
                sampled_char = ' '
            translate.append(sampled_char)
            answer.append(answer_char)
        print('---')
        print('Translation:', ''.join(translate))
        print('Answer is:', ''.join(answer))
        print('---New---')
        time.sleep(3)