from __future__ import unicode_literals, print_function, division



from io import open
import random
import os
import numpy as np

import torch
import torch.nn as nn
from torch import optim
from torch.utils.tensorboard import SummaryWriter
import torch.nn.functional as F
import time
import math
import glob
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from torchsummary import summary

#### set random_seed_number ###
random_seed_number = 1

def mse(y,t) :
    return np.sqrt((1/2)*np.mean((y-t)**2))

def gettestACC(y,t) :
    sum = 0
    for idx in range(14):
        sum += mse(y[idx, :], t[idx, :])
    return sum/14

import pickle

class Data():
    def __init__(self,x_data,y_data):
        self.x_data = x_data
        self.y_data = y_data

def synctime(inputdata,outputdata,starttime,videofs,emgfs) :
    import matplotlib.pyplot as plt

    plt.plot(inputdata[:starttime*emgfs,0],inputdata[:starttime*emgfs,1])
    plt.plot(outputdata[14,:starttime * videofs], outputdata[0,:starttime * videofs])
    plt.show()


# def dataloader(iter,inputdata,outputdata,time_length) :
#
#     index = np.random.randint(inputdata.shape[1]-time_length)
#
#     assert inputdata.shape[0] == 4 and outputdata.shape[0]==14
#     assert inputdata.shape[1] == outputdata.shape[1]
#
#     #endindex =  time_length*(iter+1) if time_length*(iter+1) < inputdata.shape[1] else inputdata.shape[1]
#     endindex = index + time_length
#     input_emg = torch.tensor(np.transpose(inputdata[:, index: endindex]), dtype=torch.float32,device=device)  # batchsize
#     target_finger_angles = torch.tensor(np.transpose(outputdata[:,index : endindex]),dtype = torch.float32,device=device) #batchsize
#     #input_emg = torch.tensor(np.transpose(inputdata[:,time_length*iter : endindex]),dtype = torch.float32,device=device) #batchsize
#     #target_finger_angles = torch.tensor(np.transpose(outputdata[:,time_length*iter : endindex]),dtype = torch.float32,device=device) #batchsize
#     #input_emg = torch.tensor(inputdata[:,iter], device = device)
#     #target_fingle_angles = torch.tensor(outputdata[:,iter] , device = device)
#     return (input_emg,target_finger_angles)

def testdataloader(timelength,inputdata,outputdata) :
    index = 0
    inputTensorList,targetTensorList = [],[]
    while index+timelength <= inputdata.shape[1] :
        inputTensor=torch.tensor(np.transpose(inputdata[:, index : index+timelength]), #(4,timelength)
                                    dtype=torch.float32, device=device)
        inputTensor = torch.unsqueeze(torch.unsqueeze(inputTensor, 1), 1)
        targetTensor = torch.tensor(np.transpose(outputdata[:, index : index+timelength]),
            dtype=torch.float32, device=device)  # (bs,4)
        targetTensor = torch.unsqueeze(torch.unsqueeze(targetTensor, 1), 1)  # (1,1,bs,14)

        inputTensorList.append(inputTensor)
        targetTensorList.append(targetTensor)

        index = index + timelength
    return inputTensorList, targetTensorList

def dataloader(iter,timelength,inputdata,outputdata,randomindex,batchsize) :
    #inputdata : (4,data_legnth)
    #outdata :  (14,data_legnth)
    assert inputdata.shape[0] ==4
    assert outputdata.shape[0] == 14
    assert inputdata.shape[1] == outputdata.shape[1]

    input_tensor_group, target_tensor_group = None, None
    for idx in range(timelength):
        if batchsize * (iter + 1) > input_data.shape[1]-timelength:
            indexend = -1
        else:
            indexend = batchsize * (iter + 1)

        input_tensor = torch.tensor(np.transpose(inputdata[:, [x+idx for x in randomindex[batchsize * iter:indexend]]]),
                                    dtype=torch.float32, device=device) #(bs,4)
        input_tensor = torch.unsqueeze(torch.unsqueeze(input_tensor, 0), 0) #(1,1,bs,4)


        target_tensor = torch.tensor(
            np.transpose(outputdata[:, [x + idx for x in randomindex[batchsize * iter:indexend]]]),
            dtype=torch.float32, device=device)  # (bs,4)
        target_tensor = torch.unsqueeze(torch.unsqueeze(target_tensor,0), 0) #(1,1,bs,14)


        if idx == 0 :
            input_tensor_group = input_tensor
            target_tensor_group = target_tensor
        else :
            input_tensor_group = torch.cat((input_tensor_group,input_tensor),dim=0) #(timelength,1,bs,14)
            target_tensor_group = torch.cat((target_tensor_group,target_tensor),dim=0) #(timelength,1,bs,14)

    if indexend != - 1:
        assert input_tensor.shape[2] == batchsize

    return input_tensor_group,target_tensor_group

def dataprepare(datapath,doesEval=False,test = False) :

    emglist, anglelist = None , None
    for filepath in glob.glob(os.path.join(datapath,'*.pkl')):
        print(filepath)
        with open(filepath,'rb') as f:
            data = pickle.load(f)
            if emglist == None and anglelist == None :
                emglist = data.x_data
                anglelist = data.y_data
            else :
                emglist.extend(data.x_data)
                anglelist.extend(data.y_data)
        if test :
            print("test on a single experiment data")
            break

    assert len(emglist) == len(anglelist)

    random.Random(random_seed_number).shuffle(emglist)
    random.Random(random_seed_number).shuffle(anglelist)

    emgarray,anglearray= None, None
    emgarray_eval,anglearray_eval = None,None
    for idx,(emg,angle) in enumerate(zip(emglist,anglelist)):
        if idx == 0 :
            emgarray = emg
            anglearray= angle

        elif idx >= len(emglist)-5 :
            if idx == len(emglist)-5 :
                emgarray_eval = emg
                anglearray_eval = angle
            else :
                emgarray_eval = np.concatenate((emgarray_eval, emg), axis=1)
                anglearray_eval = np.concatenate((anglearray_eval, angle), axis=1)
        else :
            emgarray = np.concatenate((emgarray,emg),axis = 1)
            anglearray = np.concatenate((anglearray, angle), axis=1)

    if not doesEval :
        emgarray = np.concatenate((emgarray_eval, emgarray), axis=1)
        anglearray = np.concatenate((anglearray_eval, anglearray), axis=1)
        emgarray_eval = None
        anglearray_eval = None


    return emgarray, anglearray ,emgarray_eval , anglearray_eval


class EncoderRNN(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(EncoderRNN, self).__init__()
        self.hidden_size = hidden_size
        #self.batch_size = batch_size
        self.gru = nn.GRU(input_size, hidden_size)

    def forward(self, input, hidden):
        #input : (1,bs,4)
        #hidden : (1,bs,hs)
        output, hidden = self.gru(input, hidden)
        return output, hidden #output : (1,bs,hs) , hidden : (1,bs,hs)

    #def initHidden(self):
    #    return torch.zeros(1, self.batch_size, self.hidden_size, device=device)


class AttnDecoderRNN(nn.Module):
    def __init__(self, hidden_size, output_size, dropout_p, time_length) :
        super(AttnDecoderRNN, self).__init__()
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.dropout_p = dropout_p
        self.time_length = time_length

        self.attn = nn.Linear(self.hidden_size * 2,  self.time_length)
        self.attn_combine = nn.Linear(self.hidden_size * 2, self.hidden_size)
        self.embedding = nn.Linear(self.output_size, self.hidden_size)
        self.dropout = nn.Dropout(self.dropout_p)
        self.gru = nn.GRU(self.hidden_size, self.hidden_size)
        self.out = nn.Linear(self.hidden_size, self.output_size)

    def forward(self, input, hidden, encoder_outputs):
        #input : (outputsize) / (1,bs,os)
        #hidden : (1,1,hs) / (1,bs,hs)
        #encoder_outputs : (1,1,hs) / (time_length,bs,hs)

        embedded = self.embedding(input) #embedded : (1,1,hs) / (1,bs,hs)
        embedded = self.dropout(embedded)

        attn_weights = F.softmax(self.attn(torch.cat((embedded[0], hidden[0]), 1)), dim=1)
        # embedded[0] : (1,hs) / (bs,hs) #hidden[0] : (1,hs) / (bs,hs) #torch.cat() : (1,2*hs) / (bs,hs)
        # softmax's dim =1 means applying soft max over "2*hs"
        # attn_weights : (1,time_length) / (bs,time_length)
        attn_applied = torch.bmm(attn_weights.unsqueeze(1),encoder_outputs.permute(1,0,2))
        # attn_weights.unsqueeze(0) : (1,time_length) -> (1,1,time_length) / (bs,time_length) -> (bs,1,time_length)
        # encoder_outputs.unsqueeze(1) : (time_length,hs) -> (1,time_length,hs) / (time_length,bs,hs) -> (bs,time_length,hs)
        # attn_applied : (bs,1,hs)
        output = torch.cat((embedded[0], attn_applied.squeeze(1)), 1)
        # output : (1,2*hs) / (bs,2*hs)
        output = self.attn_combine(output).unsqueeze(0)
        # output : (1,1,hs) / (1,bs,hs)

        output = F.relu(output)  # output : (1,1,hs) / (1,bs,hs)
        output, hidden = self.gru(output, hidden)
        # output : (1,bs,hs) , hidden: (1,bs,hs)
        output = self.out(output[0])
        # output : (bs,os)
        return output, hidden, attn_weights

    def initHidden(self):
        return torch.zeros(1, 1, self.hidden_size, device=device)


def train(input_tensor, target_tensor, time_length, encoder, decoder, encoder_optimizer, decoder_optimizer, criterion,tfr_prob_list,iter):
    #input_tensor : (time_length,1,bs,4)
    #target_tensor : (time_length,1,bs,14)

    encoder_hidden = torch.zeros(1, input_tensor.shape[2], encoder.hidden_size, device=device)#encoder.initHidden()

    encoder_optimizer.zero_grad()
    decoder_optimizer.zero_grad()

    assert time_length == input_tensor.shape[0] #3
    assert time_length == target_tensor.shape[0] #3
    assert input_tensor.shape[2] == input_tensor.shape[2] #batchsize
    target_size = target_tensor.shape[3]

    encoder_outputs = torch.zeros(time_length, input_tensor.shape[2],encoder.hidden_size, device=device)
    decoder_input = torch.zeros(1,input_tensor.shape[2],target_size,device=device)
    loss = 0

    for ei in range(time_length):
        encoder_output, encoder_hidden = encoder(input_tensor[ei], encoder_hidden)
        encoder_outputs[ei] = encoder_output[0]

        # input_tensor = torch.Size([length,4])  / (time_length,1,bs,4)
        # input_tensor[ei] = torch.Size([4])  / (1,bs,4)
        # encoder_hidden = torch.Size([1,1,hs]) / (1,bs,hs)
        # encoder_output = torch.Size([1,1,hs]) / (1,bs,hs)
        # encoder_outputs = torch.Size([length,hs]) / (time_length,bs,hs)

    decoder_hidden = encoder_hidden
    #decoder_hidden = torch.Size([1, bs, hs])


    prob = tfr_prob_list[iter]
    use_teacher_forcing = True if prob < teacher_forcing_ratio else False

    if use_teacher_forcing:
        # Teacher forcing: Feed the target as the next input
        for di in range(time_length):
            decoder_output, decoder_hidden, decoder_attention = decoder(decoder_input, decoder_hidden, encoder_outputs)
            # decoder_input : (1,bs,outputsize)
            # decoder_hidden : (1,bs,hs)
            # decoder_output : torch.Size([1,angle_num = 14]) / (bs,os)
            # decoder_hidden : torch.Size([1,1,hs]) / (1,bs,hs)
            # decoder_attention : torch.Size([1,max_length=10])
            loss += criterion(decoder_output, target_tensor[di].squeeze(0))
            decoder_input = target_tensor[di]  # Teacher forcing

    else:
        # Without teacher forcing: use its own predictions as the next input
        for di in range(time_length):
            decoder_output, decoder_hidden, decoder_attention = decoder(decoder_input, decoder_hidden, encoder_outputs)
            decoder_input = decoder_output.unsqueeze(0)  # detach from history as input

            loss += criterion(decoder_output, target_tensor[di].squeeze(0))


    loss.backward()

    encoder_optimizer.step()
    decoder_optimizer.step()

    return loss.item() / time_length

def trainIters(input_data,output_data,input_data_eval,output_data_eval,time_length, encoder, decoder, n_epochs, print_every, plot_every, learning_rate,batch_size):
    #input_data : (4,data_length)
    #output_data  : (14,data_length)
    #time_length = 19
    test_path = '/home/hyuninlee/PycharmProjects/xcorps/seq2seq_attentionmodel/testData/lhi'
    best_test_mse = 100
    from scipy.signal import savgol_filter
    test_input_data, test_output_data,_,_ = dataprepare(test_path, test=True)

    # test_output_data convert 0 <-> 1
    test_output_data = 1-test_output_data

    start = time.time()
    plot_losses = []
    print_loss_total = 0  # Reset every print_every
    plot_loss_total = 0  # Reset every plot_every

    encoder_optimizer = optim.Adam(encoder.parameters(), lr=learning_rate)
    decoder_optimizer = optim.Adam(decoder.parameters(), lr=learning_rate)
    criterion = nn.MSELoss()


    for epoch in range(1, n_epochs + 1):
        print('========== epoch : %d =========='% (epoch))
        ########trainstart############
        encoder.train()
        decoder.train()
        ########trainstart############
        randomindex = [x for x in range(input_data.shape[1]-time_length)]
        random.Random(epoch+random_seed_number).shuffle(randomindex)
        num_iters = (input_data.shape[1]-time_length)//batch_size

        tfr_prob_list = np.random.random(num_iters)

        for iter in range(num_iters):
            input_tensor, target_tensor = dataloader(iter, time_length, input_data, output_data, randomindex,batchsize=batch_size)

            #np.random.seed(epoch+random_seed_number)
            ## 이게 왜 있지???

            loss = train(input_tensor, target_tensor, time_length, encoder,decoder, encoder_optimizer, decoder_optimizer, criterion, tfr_prob_list,iter)
            writer.add_scalar('Loss/iter',loss,(epoch-1)*num_iters + iter)
            print_loss_total += loss
            plot_loss_total += loss

            if iter % int(0.3*(input_data.shape[1]-time_length)//batch_size) == 0 :
                print('iter : %d , loss : %.9f' % (iter, loss))


        writer.add_scalar('Loss/epoch', print_loss_total, epoch)
        ########eval start##########
        encoder.eval()
        decoder.eval()
        ########eval start##########
        if epoch % print_every == 0:
            print_loss_avg = print_loss_total / (print_every * num_iters)
            print_loss_total = 0
            print('%s (%d %d%%) loss_avg : %.9f' % (timeSince(start, epoch / n_epochs),
                                         epoch, epoch / n_epochs * 100, print_loss_avg))
            _, eval_loss_avg,eval_decoder_attention = test(input_data_eval, output_data_eval, time_length, encoder, decoder)
            writer.add_scalar('Loss/eval', eval_loss_avg, epoch)

        if epoch % plot_every == 0:
            plot_loss_avg = plot_loss_total / plot_every
            plot_losses.append(plot_loss_avg)
            plot_loss_total = 0

        test_pred_target, _,test_decoder_attention = test(test_input_data, test_output_data, time_length, encoder1, attn_decoder1)
        test_mse = gettestACC(test_pred_target,test_output_data)
        print("current test mse : %.3f" %(test_mse))
        print("best test mse : %.3f" % (best_test_mse))
        writer.add_scalar('bestTestAcc/epoch', best_test_mse, epoch)
        print('=======================================')
        if test_mse < best_test_mse :
            best_test_mse = test_mse
            print("new test mse : %.3f" %(best_test_mse))
            torch.save(encoder1.state_dict(), model_path + name + '_encoder')
            torch.save(attn_decoder1.state_dict(), model_path + name + '_attention_decoder')
            print('save model and eval attention and test attention!')
            np.save(save_path + name + '/test_pred_target.npy', test_pred_target)
            np.save(save_path + name + '/eval_decoder_attention.npy', eval_decoder_attention)
            np.save(save_path + name + '/test_decoder_attention.npy', test_decoder_attention)
            print('save target npy')




    #showPlot(plot_losses)

def test(input_data, output_data, time_length, encoder, decoder):

    loss_list = []
    criterion = nn.MSELoss()
    loss = 0
    input_tensor_list, target_tensor_list = testdataloader(time_length,input_data,output_data)
    predict_target_tensor = np.zeros_like(output_data)

    decoder_attentions = np.zeros((time_length,output_data.shape[1]))

    with torch.no_grad() :
        for idx,(input_tensor , target_tensor) in enumerate(zip(input_tensor_list,target_tensor_list)):
            encoder_hidden = torch.zeros(1, input_tensor.shape[2], encoder.hidden_size, device=device)#encoder.initHidden()

            assert time_length == input_tensor.shape[0] #3
            assert time_length == target_tensor.shape[0] #3
            assert input_tensor.shape[2] == input_tensor.shape[2] #batchsize
            target_size = target_tensor.shape[3]

            encoder_outputs = torch.zeros(time_length, input_tensor.shape[2],encoder.hidden_size, device=device)
            decoder_input = torch.zeros(1,input_tensor.shape[2],target_size,device=device)

            for ei in range(time_length):
                encoder_output, encoder_hidden = encoder(input_tensor[ei], encoder_hidden)
                encoder_outputs[ei] = encoder_output[0]

            decoder_hidden = encoder_hidden

            for di in range(time_length):
                decoder_output, decoder_hidden, decoder_attention = decoder(decoder_input, decoder_hidden, encoder_outputs)
                #save predict_Target_tensor
                predict_target_tensor[:,idx*time_length+di] = np.transpose(decoder_output.cpu().numpy()).squeeze()
                #save decoder attention scores into matrix
                decoder_attentions[:,idx*time_length+di] = np.transpose(decoder_attention.cpu().numpy()).squeeze()

                decoder_input = decoder_output.unsqueeze(0)  # detach from history as input
                loss += criterion(decoder_output, target_tensor[di].squeeze(0))


            #writer.add_scalar('Loss/test ', loss.item()/time_length, iter)
            loss_list.append(loss.item()/time_length)
    loss_avg = sum(loss_list)/len(loss_list)
    print("eval loss : %.9f " %(loss_avg))
    return predict_target_tensor , loss_avg , decoder_attentions

def asMinutes(s):
    m = math.floor(s / 60)
    s -= m * 60
    return '%dm %ds' % (m, s)


def timeSince(since, percent):
    now = time.time()
    s = now - since
    es = s / (percent)
    rs = es - s
    return '%s (- %s)' % (asMinutes(s), asMinutes(rs))

def showPlot(points):
    plt.figure()
    plt.plot(points)
    plt.show()


def ComparePlotAngles(truthAngle,PredAngle):
    fig2 = plt.figure(2)
    fig2.suptitle('Thumb',fontsize=40)
    ax1 = fig2.add_subplot(211)
    ax2 = fig2.add_subplot(212)
    #ax1.set_title('Thumb',fontsize=20)
    ax1.plot(truthAngle[0, :], color = 'k', label='truth')
    ax2.plot(truthAngle[1, :], color = 'k', label='truth')
    ax1.plot(PredAngle[0, :], color = 'r',label='prediction')
    ax2.plot(PredAngle[1, :], color = 'r',label='prediction')
    ax1.set_ylabel('MCP ',fontsize=20)
    ax2.set_ylabel('PIP',fontsize=20)
    ax1.legend(loc='upper right')
    ax2.legend(loc='upper right')

    fig3 = plt.figure(3)
    fig3.suptitle('Index', fontsize=40)
    ax1 = fig3.add_subplot(311)
    ax2 = fig3.add_subplot(312)
    ax3 = fig3.add_subplot(313)
    #ax1.set_title('index',fontsize=20)
    ax1.plot(truthAngle[2, :], color = 'k', label='truth')
    ax2.plot(truthAngle[3, :], color = 'k', label='truth')
    ax3.plot(truthAngle[4, :],color = 'k', label='truth')
    ax1.plot(PredAngle[2, :], color = 'r',label='prediction')
    ax2.plot(PredAngle[3, :], color = 'r',label='prediction')
    ax3.plot(PredAngle[4, :],color = 'r',label='prediction')
    ax1.set_ylabel('MCP ',fontsize=20)
    ax2.set_ylabel('PIP',fontsize=20)
    ax3.set_ylabel('DIP', fontsize=20)
    ax1.legend(loc='upper right')
    ax2.legend(loc='upper right')
    ax3.legend(loc='upper right')


    fig4 = plt.figure(4)
    fig4.suptitle('Middle', fontsize=40)
    ax1 = fig4.add_subplot(311)
    ax2 = fig4.add_subplot(312)
    ax3 = fig4.add_subplot(313)
    #ax1.title.set_text('middle',fontsize=20)
    ax1.plot(truthAngle[5, :], color = 'k', label='truth')
    ax2.plot(truthAngle[6, :], color = 'k', label='truth')
    ax3.plot(truthAngle[7, :], color = 'k', label='truth')
    ax1.plot(PredAngle[5, :],  color = 'r',label='prediction')
    ax2.plot(PredAngle[6, :],  color = 'r',label='prediction')
    ax3.plot(PredAngle[7, :],  color = 'r',label='prediction')
    ax1.set_ylabel('MCP ',fontsize=20)
    ax2.set_ylabel('PIP',fontsize=20)
    ax3.set_ylabel('DIP', fontsize=20)
    ax1.legend(loc='upper right')
    ax2.legend(loc='upper right')
    ax3.legend(loc='upper right')


    fig5 = plt.figure(5)
    fig5.suptitle('Ring', fontsize=40)
    ax1 = fig5.add_subplot(311)
    ax2 = fig5.add_subplot(312)
    ax3 = fig5.add_subplot(313)
    #ax1.title.set_text('ring',fontsize=20)
    ax1.plot(truthAngle[8, :], color = 'k', label='truth')
    ax2.plot(truthAngle[9, :], color = 'k', label='truth')
    ax3.plot(truthAngle[10, :], color = 'k', label='truth')
    ax1.plot(PredAngle[8, :], color = 'r',label='prediction')
    ax2.plot(PredAngle[9, :], color = 'r',label='prediction')
    ax3.plot(PredAngle[10, :], color = 'r',label='prediction')
    ax1.set_ylabel('MCP ',fontsize=20)
    ax2.set_ylabel('PIP',fontsize=20)
    ax3.set_ylabel('DIP', fontsize=20)
    ax1.legend(loc='upper right')
    ax2.legend(loc='upper right')
    ax3.legend(loc='upper right')

    fig6 = plt.figure(6)
    fig6.suptitle('Pinky', fontsize=40)
    ax1 = fig6.add_subplot(311)
    ax2 = fig6.add_subplot(312)
    ax3 = fig6.add_subplot(313)
    #ax1.title.set_text('pinky',fontsize=20)
    ax1.plot(truthAngle[11, :], color = 'k', label='truth')
    ax2.plot(truthAngle[12, :], color = 'k', label='truth')
    ax3.plot(truthAngle[13, :], color = 'k', label='truth')
    ax1.plot(PredAngle[11, :], color = 'r',label='prediction')
    ax2.plot(PredAngle[12, :], color = 'r',label='prediction')
    ax3.plot(PredAngle[13, :], color = 'r',label='prediction')
    ax1.set_ylabel('MCP ',fontsize=20)
    ax2.set_ylabel('PIP',fontsize=20)
    ax3.set_ylabel('DIP', fontsize=20)
    ax1.legend(loc='upper right')
    ax2.legend(loc='upper right')
    ax3.legend(loc='upper right')

    plt.show()


if __name__=="__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # fixed hyperparameters
    data_path = '/home/hyuninlee/PycharmProjects/xcorps/seq2seq_attentionmodel/trainData/lhi/2sec'
    test_path = '/home/hyuninlee/PycharmProjects/xcorps/seq2seq_attentionmodel/testData/lhi'
    save_path = '/home/hyuninlee/PycharmProjects/xcorps/seq2seq_attentionmodel/result/Attention_old_2/'
    model_path = '/home/hyuninlee/PycharmProjects/xcorps/seq2seq_attentionmodel/result/Attention_old_2/'
    assert save_path == model_path
    N_emgsensor = 4
    N_fingerAngle = 14


    #set torch random seed number
    torch.manual_seed(random_seed_number)


    #variable hyperparameters
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--hidden_size', type=int, default=256)
    parser.add_argument('--teacher_forcing_ratio','-tfr' ,type=float, default=0.7)
    parser.add_argument('--train_iter','-ti' ,type=int, default=5)
    parser.add_argument('--learning_rate','-lr' ,type=float, default=0.001)
    parser.add_argument('--time_length','-tl' ,type=int, default=19)
    parser.add_argument('--batch_size', type=int, default=1024)
    parser.add_argument('--doesTest', type=bool, default=True)
    parser.add_argument('--doesEval', type=bool, default=True)
    args = parser.parse_args()

    params = vars(args) #convert args to dirctionary

    hidden_size = params['hidden_size']
    teacher_forcing_ratio = params['teacher_forcing_ratio']
    train_iter = params['train_iter']
    learning_rate = params['learning_rate']
    time_length = params['time_length']
    batch_size = params['batch_size']
    doesTest = params['doesTest']
    doesEval = params['doesEval']

    ## set random seed

    """
    1. randomly mix traindata_lhi_1 ,2,3 
    2. randomly pick 1 motion
    3. random teacher forcing
    """
    name = 'lr_' + str(learning_rate)+'_bs_'+str(batch_size)+'_tfr_'+str(teacher_forcing_ratio)+'_tl_'+str(time_length)+'__'+time.strftime("%d-%m-%Y_%H-%M-%S")

    writer = SummaryWriter(log_dir = save_path+name)

    #prpare data
    input_data, output_data,input_data_eval,output_data_eval = dataprepare(data_path,doesEval)

    #model
    encoder1 = EncoderRNN(N_emgsensor, hidden_size).to(device)
    attn_decoder1 = AttnDecoderRNN(hidden_size, N_fingerAngle, dropout_p=0.1,time_length=time_length).to(device)
    #summary(encoder1 , [(batch_size, N_emgsensor),(batch_size, hidden_size)])

    #print parameter nyumber of encoder1 and attn_decoder1
    model_parameters = filter(lambda p: p.requires_grad, encoder1.parameters())
    params = sum([np.prod(p.size()) for p in model_parameters])
    print(params)
    model_parameters = filter(lambda p: p.requires_grad, attn_decoder1.parameters())
    params = sum([np.prod(p.size()) for p in model_parameters])
    print(params)

    #train
    train_param = {
        "innput_data" : input_data,
        "output_data" : output_data,
        "time_length" : time_length,
        "encoder" : encoder1,
        "decoder" : attn_decoder1,
        "n_epochs" : train_iter,
        "print_every" : 1,
        "plot_every" : 1,
        "learning_rate" : learning_rate,
        "batchsize" : batch_size,
    }
    trainIters(input_data, output_data,input_data_eval,output_data_eval,time_length, encoder1, attn_decoder1, train_iter, print_every=1, plot_every=1, learning_rate=learning_rate,batch_size = batch_size)


    writer.flush()
    writer.close()

    # if doesTest :
    #     from scipy.signal import savgol_filter
    #     test_input_data, test_output_data,_,_ = dataprepare(test_path, test=True)
    #
    #     # test_output_data convert 0 <-> 1
    #     test_output_data = 1-test_output_data
    #
    #     test_pred_target,_ = test(test_input_data,test_output_data,time_length, encoder1, attn_decoder1)
    #     ComparePlotAngles(test_output_data,test_pred_target)
    #
    #     np.save(save_path+name+'/test_pred_target.npy', test_pred_target)
    #
    #     smooth_test_pred_target = np.zeros_like(test_pred_target)
    #     for i in range(14) :
    #         smooth_test_pred_target[i, :] = savgol_filter(test_pred_target[i, :], 59, 5)
    #     ComparePlotAngles(smooth_test_pred_target, test_output_data)
    #
    #     for i in range(14):
    #         smooth_test_pred_target[i, :] = savgol_filter(test_pred_target[i, :], 99, 5)
    #     ComparePlotAngles(smooth_test_pred_target, test_output_data)
    #
    #     for i in range(14):
    #         smooth_test_pred_target[i, :] = savgol_filter(test_pred_target[i, :], 159, 5)
    #     ComparePlotAngles(smooth_test_pred_target, test_output_data)
    #
    #     for i in range(14):
    #         smooth_test_pred_target[i, :] = savgol_filter(test_pred_target[i, :], 199, 5)
    #     ComparePlotAngles(smooth_test_pred_target, test_output_data)


