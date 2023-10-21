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

from tqdm import tqdm

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
    fingertype = [i+1 for i in range(5)]*5*9
    random.Random(0).shuffle(emglist)
    random.Random(0).shuffle(anglelist)
    random.Random(0).shuffle(fingertype)

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
    def __init__(self, hidden_size, output_size, dropout_p, decoder_time_length) :
        super(AttnDecoderRNN, self).__init__()
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.dropout_p = dropout_p
        self.decoder_time_length = decoder_time_length

        self.attn = nn.Linear(self.hidden_size * 2,  self.decoder_time_length)
        self.attn_combine = nn.Linear(self.hidden_size * 2, self.hidden_size)
        self.embedding = nn.Linear(self.output_size, self.hidden_size)
        self.dropout = nn.Dropout(self.dropout_p)
        self.gru = nn.GRU(self.hidden_size, self.hidden_size)
        self.out = nn.Linear(self.hidden_size, self.output_size)

    def forward(self, input, hidden, encoder_outputs):
        #input : (outputsize) / (1,bs,os)
        #hidden : (1,1,hs) / (1,bs,hs)
        #encoder_outputs : (1,1,hs) / (time_length=4,bs,hs)

        embedded = self.embedding(input) #embedded : (1,1,hs) / (1,bs,hs)
        embedded = self.dropout(embedded)

        attn_weights = F.softmax(self.attn(torch.cat((embedded[0], hidden[0]), 1)), dim=1)
        # embedded[0] : (1,hs) / (bs,hs) #hidden[0] : (1,hs) / (bs,hs) #torch.cat() : (1,2*hs) / (bs,hs)
        # softmax's dim =1 means applying soft max over "2*hs"
        # attn_weights : (1,decoder_time_length=4) / (bs,decoder_time_length=4)
        attn_applied = torch.bmm(attn_weights.unsqueeze(1),encoder_outputs.permute(1,0,2))
        # attn_weights.unsqueeze(0) : (1,decoder_time_length) -> (1,1,decoder_time_length) / (bs,decoder_time_length) -> (bs,1,decoder_time_length)
        # encoder_outputs.unsqueeze(1) : (time_length=4,hs) -> (1,time_length=4,hs) / (time_length=4,bs,hs) -> (bs,time_length=4,hs)
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


def train(input_tensor, target_tensor, time_length, encoder, decoder_thumb, decoder_index,decoder_middle,
                         decoder_ring,decoder_pinky,encoder_optimizer, decoder_thumb_optimizer, decoder_index_optimizer,
                         decoder_middle_optimizer,decoder_ring_optimizer,decoder_pinky_optimizer, criterion,tfr_prob_list,iter):
    #input_tensor : (time_length=1,1,bs,4)
    #target_tensor : (time_length=1,1,bs,14)

    assert input_tensor.shape[2] == input_tensor.shape[2] #batchsize
    assert input_tensor.shape[0] == 1
    assert target_tensor.shape[0] == 1
    new_input_tensor = torch.zeros(input_tensor.shape[3],input_tensor.shape[1],input_tensor.shape[2],input_tensor.shape[3], device=device)
    new_target_tensor = torch.zeros(target_tensor.shape[3], target_tensor.shape[1], target_tensor.shape[2],target_tensor.shape[3], device=device)
    #should convert (1,1,bs,4) to (4,1,bs,4) as one hot vector
    for idx in range(input_tensor.shape[3]) :
        new_input_tensor[idx,:,:,idx] = input_tensor[0,0,:,idx]
    for idx in range(target_tensor.shape[3]):
        new_target_tensor[idx,:,:,idx] = target_tensor[0,0,:,idx]

    del input_tensor
    del target_tensor


    # now feed encoder new_input_tensor[0,:,:,:] ~ new_input_tensor[3,:,:,:] to encoder
    # new_input_tensor[idx] has same size with input_tensor[idx] as [1,bs,4]
    encoder_hidden = torch.zeros(1, new_input_tensor.shape[2], encoder.hidden_size, device=device)#encoder.initHidden()
    encoder_optimizer.zero_grad()

    decoder_thumb_optimizer.zero_grad()
    decoder_index_optimizer.zero_grad()
    decoder_middle_optimizer.zero_grad()
    decoder_ring_optimizer.zero_grad()
    decoder_pinky_optimizer.zero_grad()


    encoder_outputs = torch.zeros(N_emgsensor, new_input_tensor.shape[2],encoder.hidden_size, device=device)

    loss_total,loss_thumb,loss_index,loss_middle,loss_ring,loss_pinky = 0,0,0,0,0,0

    for ei in range(N_emgsensor):
        encoder_output, encoder_hidden = encoder(new_input_tensor[ei], encoder_hidden)
        encoder_outputs[ei] = encoder_output[0]

        # new_input_tensor = torch.Size([length,4])  / (time_length=N_emgsensor=4,1,bs,4)
        # input_tensor[ei] = torch.Size([4])  / (1,bs,4)
        # encoder_hidden = torch.Size([1,1,hs]) / (1,bs,hs)
        # encoder_output = torch.Size([1,1,hs]) / (1,bs,hs)
        # encoder_outputs = torch.Size([length,hs]) / (time_length=N_emgsensor=4,bs,hs)

    decoder_input_thumb = torch.zeros(1, new_input_tensor.shape[2], 2, device=device)
    decoder_input_index = torch.zeros(1, new_input_tensor.shape[2], 3, device=device)
    decoder_input_middle = torch.zeros(1, new_input_tensor.shape[2], 3, device=device)
    decoder_input_ring = torch.zeros(1, new_input_tensor.shape[2], 3, device=device)
    decoder_input_pinky = torch.zeros(1, new_input_tensor.shape[2], 3, device=device)

    decoder_hidden_thumb = encoder_hidden
    decoder_hidden_index = encoder_hidden
    decoder_hidden_middle = encoder_hidden
    decoder_hidden_ring = encoder_hidden
    decoder_hidden_pinky = encoder_hidden

    #decoder_hidden = torch.Size([1, bs, hs])

    prob = tfr_prob_list[iter]
    use_teacher_forcing = True if prob < teacher_forcing_ratio else False

    if use_teacher_forcing:
        # Teacher forcing: Feed the target as the next input
        for di in range(0,2):
            decoder_output_thumb, decoder_hidden_thumb, decoder_attention_thumb = decoder_thumb(decoder_input_thumb, decoder_hidden_thumb,
                                                                              encoder_outputs)
            # decoder_input : (1,bs,os)
            # decoder_hidden : (1,bs,hs)
            # decoder_output : torch.Size([1,angle_num = 14]) / (bs,os)
            # decoder_hidden : torch.Size([1,1,hs]) / (1,bs,hs)
            # decoder_attention : torch.Size([1,max_length=10])
            loss_thumb += criterion(decoder_output_thumb, new_target_tensor[di,:,:,0:2].squeeze(0))
            decoder_input_thumb = new_target_tensor[di,:,:,0:2]  # Teacher forcing

        for di in range(2,5):
            decoder_output_index, decoder_hidden_index, decoder_attention_index = decoder_index(decoder_input_index, decoder_hidden_index,
                                                                              encoder_outputs)
            # decoder_input : (1,bs,os)
            # decoder_hidden : (1,bs,hs)
            # decoder_output : torch.Size([1,angle_num = 14]) / (bs,os)
            # decoder_hidden : torch.Size([1,1,hs]) / (1,bs,hs)
            # decoder_attention : torch.Size([1,max_length=10])
            loss_index += criterion(decoder_output_index, new_target_tensor[di,:,:,2:5].squeeze(0))
            decoder_input_index = new_target_tensor[di,:,:,2:5]  # Teacher forcing

        for di in range(5,8):
            decoder_output_middle, decoder_hidden_middle, decoder_attention_middle = decoder_middle(decoder_input_middle, decoder_hidden_middle,
                                                                              encoder_outputs)
            # decoder_input : (1,bs,os)
            # decoder_hidden : (1,bs,hs)
            # decoder_output : torch.Size([1,angle_num = 14]) / (bs,os)
            # decoder_hidden : torch.Size([1,1,hs]) / (1,bs,hs)
            # decoder_attention : torch.Size([1,max_length=10])
            loss_middle += criterion(decoder_output_middle, new_target_tensor[di,:,:,5:8].squeeze(0))
            decoder_input_middle = new_target_tensor[di,:,:,5:8]  # Teacher forcing

        for di in range(8,11):
            decoder_output_ring, decoder_hidden_ring, decoder_attention_ring = decoder_ring(decoder_input_ring, decoder_hidden_ring,
                                                                              encoder_outputs)
            # decoder_input : (1,bs,os)
            # decoder_hidden : (1,bs,hs)
            # decoder_output : torch.Size([1,angle_num = 14]) / (bs,os)
            # decoder_hidden : torch.Size([1,1,hs]) / (1,bs,hs)
            # decoder_attention : torch.Size([1,max_length=10])
            loss_ring += criterion(decoder_output_ring, new_target_tensor[di,:,:,8:11].squeeze(0))
            decoder_input_ring = new_target_tensor[di,:,:,8:11]  # Teacher forcing

        for di in range(11,14):
            decoder_output_pinky, decoder_hidden_pinky, decoder_attention_pinky = decoder_pinky(decoder_input_pinky, decoder_hidden_pinky,
                                                                              encoder_outputs)
            # decoder_input : (1,bs,os)
            # decoder_hidden : (1,bs,hs)
            # decoder_output : torch.Size([1,angle_num = 14]) / (bs,os)
            # decoder_hidden : torch.Size([1,1,hs]) / (1,bs,hs)
            # decoder_attention : torch.Size([1,max_length=10])
            loss_pinky += criterion(decoder_output_pinky, new_target_tensor[di,:,:,11:14].squeeze(0))
            decoder_input_pinky = new_target_tensor[di,:,:,11:14]  # Teacher forcing
    else:
        # Without teacher forcing: use its own predictions as the next input
        for di in range(0, 2):
            decoder_output_thumb, decoder_hidden_thumb, decoder_attention_thumb = decoder_thumb(decoder_input_thumb,
                                                                                                decoder_hidden_thumb,
                                                                                                encoder_outputs)
            loss_thumb += criterion(decoder_output_thumb, new_target_tensor[di,:,:,0:2].squeeze(0))
            decoder_input_thumb = decoder_output_thumb.unsqueeze(0)

        for di in range(2, 5):
            decoder_output_index, decoder_hidden_index, decoder_attention_index = decoder_index(decoder_input_index,
                                                                                                decoder_hidden_index,
                                                                                                encoder_outputs)
            loss_index += criterion(decoder_output_index, new_target_tensor[di,:,:,2:5].squeeze(0))
            decoder_input_index = decoder_output_index.unsqueeze(0)

        for di in range(5, 8):
            decoder_output_middle, decoder_hidden_middle, decoder_attention_middle = decoder_middle(
                decoder_input_middle, decoder_hidden_middle,
                encoder_outputs)

            loss_middle += criterion(decoder_output_middle, new_target_tensor[di,:,:,5:8].squeeze(0))
            decoder_input_middle = decoder_output_middle.unsqueeze(0)

        for di in range(8, 11):
            decoder_output_ring, decoder_hidden_ring, decoder_attention_ring = decoder_ring(decoder_input_ring,
                                                                                            decoder_hidden_ring,
                                                                                            encoder_outputs)

            loss_ring += criterion(decoder_output_ring, new_target_tensor[di,:,:,8:11].squeeze(0))
            decoder_input_ring = decoder_output_ring.unsqueeze(0)

        for di in range(11, 14):
            decoder_output_pinky, decoder_hidden_pinky, decoder_attention_pinky = decoder_pinky(decoder_input_pinky,
                                                                                                decoder_hidden_pinky,
                                                                                                encoder_outputs)

            loss_pinky += criterion(decoder_output_pinky, new_target_tensor[di,:,:,11:14].squeeze(0))
            decoder_input_pinky = decoder_output_pinky.unsqueeze(0)
    loss_total = loss_thumb + loss_index + loss_middle + loss_ring + loss_pinky

    loss_total.backward()

    encoder_optimizer.step()
    decoder_thumb_optimizer.step()
    decoder_index_optimizer.step()
    decoder_middle_optimizer.step()
    decoder_ring_optimizer.step()
    decoder_pinky_optimizer.step()


    return loss_total.item() / 14 , loss_thumb/2 , loss_index /3 , loss_middle /3 , loss_ring /3 , loss_pinky /3 ,

def trainIters(input_data,output_data,input_data_eval,output_data_eval,time_length, encoder, decoder_thumb,
               decoder_index,decoder_middle,decoder_ring,decoder_pinky, n_epochs, eval_every, test_every,
               learning_rate_encoder,learning_rate_decoder,batch_size):
    #input_data : (4,data_length)
    #output_data  : (14,data_length)
    #time_length = 19
    best_test_mse = 100
    best_eval_mse = 100
    from scipy.signal import savgol_filter
    test_input_data, test_output_data,_,_ = dataprepare(test_path, test=True)

    # test_output_data convert 0 <-> 1
    test_output_data = 1-test_output_data

    start = time.time()

    loss_total, loss1_total, loss2_total, loss3_total, loss4_total, loss5_total = 0, 0, 0, 0, 0, 0


    encoder_optimizer = optim.Adam(encoder.parameters(), lr=learning_rate_encoder)
    decoder_thumb_optimizer = optim.Adam(decoder_thumb.parameters(), lr=learning_rate_decoder)
    decoder_index_optimizer = optim.Adam(decoder_index.parameters(), lr=learning_rate_decoder)
    decoder_middle_optimizer = optim.Adam(decoder_middle.parameters(), lr=learning_rate_decoder)
    decoder_ring_optimizer = optim.Adam(decoder_ring.parameters(), lr=learning_rate_decoder)
    decoder_pinky_optimizer = optim.Adam(decoder_pinky.parameters(), lr=learning_rate_decoder)

    criterion = nn.MSELoss()


    for epoch in range(1, n_epochs + 1):
        print('========== epoch : %d =========='% (epoch))
        randomindex = [x for x in range(input_data.shape[1]-time_length)]
        random.Random(epoch).shuffle(randomindex)
        num_iters = (input_data.shape[1]-time_length)//batch_size

        tfr_prob_list = np.random.random(num_iters)

        for iter in range(num_iters):
            input_tensor, target_tensor = dataloader(iter, time_length, input_data, output_data, randomindex,batchsize=batch_size)
            np.random.seed(epoch)

            loss_5finger,loss1,loss2,loss3,loss4,loss5 = train(input_tensor, target_tensor, time_length, encoder,decoder_thumb, decoder_index,decoder_middle,
                         decoder_ring,decoder_pinky,encoder_optimizer, decoder_thumb_optimizer, decoder_index_optimizer,
                         decoder_middle_optimizer,decoder_ring_optimizer,decoder_pinky_optimizer,criterion, tfr_prob_list,iter)
            writer.add_scalar('Loss/iter',loss_5finger,(epoch-1)*num_iters + iter)
            loss_total += loss_5finger
            loss1_total += loss1
            loss2_total += loss2
            loss3_total += loss3
            loss4_total += loss4
            loss5_total += loss5


            if iter % int(0.3*(input_data.shape[1]-time_length)//batch_size) == 0 :
                print('iter : %d , loss_5finger : %.5f' % (iter, loss_5finger))

        loss_avg = loss_total / num_iters
        loss1_avg = loss1_total / num_iters
        loss2_avg = loss2_total / num_iters
        loss3_avg = loss3_total / num_iters
        loss4_avg = loss4_total / num_iters
        loss5_avg = loss5_total / num_iters

        writer.add_scalar('Loss_total/epoch', loss_avg, epoch)
        writer.add_scalar('Loss_thumb/epoch', loss1_avg, epoch)
        writer.add_scalar('Loss_index/epoch', loss2_avg, epoch)
        writer.add_scalar('Loss_middle/epoch', loss3_avg, epoch)
        writer.add_scalar('Loss_ring/epoch', loss4_avg, epoch)
        writer.add_scalar('Loss_pinky/epoch', loss5_avg, epoch)

        loss_total, loss1_total, loss2_total, loss3_total, loss4_total, loss5_total = 0, 0, 0, 0, 0, 0


        print('%s (%d %d%%) loss_avg : %.9f' % (timeSince(start, epoch / n_epochs),
                                     epoch, epoch / n_epochs * 100, loss_avg))

        if epoch % eval_every == 0 :
            eval_pred_target, eval_loss_avg, eval_loss1_avg, eval_loss2_avg, eval_loss3_avg, eval_loss4_avg, eval_loss5_avg, eval_attention_scores \
                = test(input_data_eval, output_data_eval, time_length, encoder, decoder_thumb, decoder_index,
                                    decoder_middle,decoder_ring,decoder_pinky)
            np.save(save_path + name + '/epoch_'+str(epoch)+'_eval_attention_scores.npy', eval_attention_scores.cpu().numpy())

            eval_mse = gettestACC(eval_pred_target,output_data_eval)
            print("current eval mse : %.3f" % (eval_mse))
            print("best eval mse : %.3f" % (best_eval_mse))
            writer.add_scalar('EvalAcc/epoch', eval_mse, epoch)
            writer.add_scalar('bestEvalAcc/epoch', best_eval_mse, epoch)
            print('=======================================')
            writer.add_scalar('Loss_total/eval', eval_loss_avg, epoch)
            writer.add_scalar('Loss_thumb/eval', eval_loss1_avg, epoch)
            writer.add_scalar('Loss_index/eval', eval_loss2_avg, epoch)
            writer.add_scalar('Loss_middle/eval', eval_loss3_avg, epoch)
            writer.add_scalar('Loss_ring/eval', eval_loss4_avg, epoch)
            writer.add_scalar('Loss_pinky/eval', eval_loss5_avg, epoch)
            if eval_mse < best_eval_mse :
                best_eval_mse = eval_mse
                print("new eval mse : %.3f" %(best_eval_mse))
                print('save eval attention and eval pred angle')
                np.save(save_path + name + '/best_eval_pred_target.npy', eval_pred_target)
                np.save(save_path + name + '/best_eval_attention_scores.npy', eval_attention_scores.cpu().numpy())

        if epoch % test_every == 0 :
            test_pred_target, test_loss_avg, test_loss1_avg, test_loss2_avg, test_loss3_avg, test_loss4_avg, test_loss5_avg, test_attention_scores \
                = test(test_input_data, test_output_data, time_length, encoder, decoder_thumb, decoder_index,
                       decoder_middle, decoder_ring, decoder_pinky)
            test_mse = gettestACC(test_pred_target,test_output_data)
            print("current test mse : %.3f" %(test_mse))
            print("best test mse : %.3f" % (best_test_mse))
            writer.add_scalar('TestAcc/epoch', test_mse, epoch)
            writer.add_scalar('bestTestAcc/epoch', best_test_mse, epoch)
            print('=======================================')
            if test_mse < best_test_mse :
                best_test_mse = test_mse
                print("new test mse : %.3f" %(best_test_mse))
                print('savemodel when best test mse!')
                torch.save(encoder.state_dict(), model_path + name + '_encoder')
                torch.save(decoder_thumb.state_dict(), model_path + name + '_attention_decoder_thumb')
                torch.save(decoder_index.state_dict(), model_path + name + '_attention_decoder_index')
                torch.save(decoder_middle.state_dict(), model_path + name + '_attention_decoder_middle')
                torch.save(decoder_ring.state_dict(), model_path + name + '_attention_decoder_ring')
                torch.save(decoder_pinky.state_dict(), model_path + name + '_attention_decoder_pinky')
                print('save test attention and test pred angle')
                np.save(save_path + name + '/best_test_pred_target.npy', test_pred_target)
                np.save(save_path + name + '/best_test_attention_scores.npy', test_attention_scores.cpu().numpy())



    #showPlot(plot_losses)

def test(input_data, output_data, time_length, encoder, decoder_thumb, decoder_index,
                                decoder_middle, decoder_ring, decoder_pinky):

    criterion = nn.MSELoss()
    loss_5finger_total, loss1_total, loss2_total, loss3_total, loss4_total, loss5_total = 0,0,0,0,0,0


    input_tensor_list, target_tensor_list = testdataloader(time_length,input_data,output_data)
    predict_target_tensor = np.zeros_like(output_data)

    #define attentinoscore
    attentionscores = torch.zeros(N_emgsensor*14,len(target_tensor_list),device=device)

    with torch.no_grad() :
        for idx0,(input_tensor , target_tensor) in enumerate(zip(input_tensor_list,target_tensor_list)):

            loss_5finger, loss_thumb, loss_index, loss_middle, loss_ring, loss_pinky = 0, 0, 0, 0, 0, 0

            assert input_tensor.shape[2] == input_tensor.shape[2]  # batchsize
            assert input_tensor.shape[0] == 1
            assert target_tensor.shape[0] == 1

            new_input_tensor = torch.zeros(input_tensor.shape[3], input_tensor.shape[1], input_tensor.shape[2],
                                           input_tensor.shape[3], device=device)
            new_target_tensor = torch.zeros(target_tensor.shape[3], target_tensor.shape[1], target_tensor.shape[2],
                                            target_tensor.shape[3], device=device)
            # should convert (1,1,bs,4) to (4,1,bs,4) as one hot vector
            for idx1 in range(input_tensor.shape[3]):
                new_input_tensor[idx1, :, :, idx1] = input_tensor[0, 0, :, idx1]
            for idx2 in range(target_tensor.shape[3]):
                new_target_tensor[idx2, :, :, idx2] = target_tensor[0, 0, :, idx2]

            del input_tensor
            del target_tensor
            # also define attention score matrix to show its effectiveness
            encoder_hidden = torch.zeros(1, new_input_tensor.shape[2], encoder.hidden_size, device=device)#encoder.initHidden()
            encoder_outputs = torch.zeros(N_emgsensor, new_input_tensor.shape[2],encoder.hidden_size, device=device)

            for ei in range(N_emgsensor):
                encoder_output, encoder_hidden = encoder(new_input_tensor[ei], encoder_hidden)
                encoder_outputs[ei] = encoder_output[0]

            decoder_input_thumb = torch.zeros(1, new_input_tensor.shape[2], 2, device=device)
            decoder_input_index = torch.zeros(1, new_input_tensor.shape[2], 3, device=device)
            decoder_input_middle = torch.zeros(1, new_input_tensor.shape[2], 3, device=device)
            decoder_input_ring = torch.zeros(1, new_input_tensor.shape[2], 3, device=device)
            decoder_input_pinky = torch.zeros(1, new_input_tensor.shape[2], 3, device=device)

            decoder_hidden_thumb = encoder_hidden
            decoder_hidden_index = encoder_hidden
            decoder_hidden_middle = encoder_hidden
            decoder_hidden_ring = encoder_hidden
            decoder_hidden_pinky = encoder_hidden


            for di in range(0, 2):
                decoder_output_thumb, decoder_hidden_thumb, decoder_attention_thumb = decoder_thumb(decoder_input_thumb,
                                                                                                    decoder_hidden_thumb,
                                                                                                    encoder_outputs)
                predict_target_tensor[di, idx0 * time_length] = np.transpose(decoder_output_thumb[0,di].cpu().numpy()).squeeze()
                loss_thumb += criterion(decoder_output_thumb, new_target_tensor[di, :, :, 0:2].squeeze(0))
                decoder_input_thumb = decoder_output_thumb.unsqueeze(0)
                #save attentionscores
                attentionscores[N_emgsensor * di:N_emgsensor * (di+1),idx0] = decoder_attention_thumb

            for di in range(2, 5):
                decoder_output_index, decoder_hidden_index, decoder_attention_index = decoder_index(decoder_input_index,
                                                                                                    decoder_hidden_index,
                                                                                                    encoder_outputs)
                predict_target_tensor[di, idx0 * time_length] = np.transpose(
                    decoder_output_index[0,di-2].cpu().numpy()).squeeze()
                loss_index += criterion(decoder_output_index, new_target_tensor[di, :, :, 2:5].squeeze(0))
                decoder_input_index = decoder_output_index.unsqueeze(0)
                # save attentionscores
                attentionscores[N_emgsensor * di:N_emgsensor * (di+1),idx0] = decoder_attention_index

            for di in range(5, 8):
                decoder_output_middle, decoder_hidden_middle, decoder_attention_middle = decoder_middle(
                    decoder_input_middle, decoder_hidden_middle,
                    encoder_outputs)
                predict_target_tensor[di, idx0 * time_length] = np.transpose(
                    decoder_output_middle[0,di-5].cpu().numpy()).squeeze()
                loss_middle += criterion(decoder_output_middle, new_target_tensor[di, :, :, 5:8].squeeze(0))
                decoder_input_middle = decoder_output_middle.unsqueeze(0)
                # save attentionscores
                attentionscores[N_emgsensor * di:N_emgsensor * (di+1),idx0] = decoder_attention_middle

            for di in range(8, 11):
                decoder_output_ring, decoder_hidden_ring, decoder_attention_ring = decoder_ring(decoder_input_ring,
                                                                                                decoder_hidden_ring,
                                                                                                encoder_outputs)
                predict_target_tensor[di, idx0 * time_length] = np.transpose(
                    decoder_output_ring[0,di-8].cpu().numpy()).squeeze()
                loss_ring += criterion(decoder_output_ring, new_target_tensor[di, :, :, 8:11].squeeze(0))
                decoder_input_ring = decoder_output_ring.unsqueeze(0)
                # save attentionscores
                attentionscores[N_emgsensor * di:N_emgsensor * (di+1),idx0] = decoder_attention_ring

            for di in range(11, 14):
                decoder_output_pinky, decoder_hidden_pinky, decoder_attention_pinky = decoder_pinky(decoder_input_pinky,
                                                                                                    decoder_hidden_pinky,
                                                                                                    encoder_outputs)
                predict_target_tensor[di, idx0 * time_length] = np.transpose(
                    decoder_output_pinky[0,di-11].cpu().numpy()).squeeze()
                loss_pinky += criterion(decoder_output_pinky, new_target_tensor[di, :, :, 11:14].squeeze(0))
                decoder_input_pinky = decoder_output_pinky.unsqueeze(0)
                # save attentionscores
                attentionscores[N_emgsensor * di:N_emgsensor * (di+1),idx0] = decoder_attention_pinky

            loss_5finger = loss_thumb + loss_index + loss_middle + loss_ring + loss_pinky
            #loss_5finger: total 14 EA loss sum

            loss1_total += loss_thumb
            loss2_total += loss_index
            loss3_total += loss_middle
            loss4_total += loss_ring
            loss5_total += loss_pinky

            loss_5finger_total += loss_5finger

        assert len(input_tensor_list) == (idx0 + 1)

        loss_avg = loss_5finger_total / ((idx0+1)*14)
        loss1_avg = loss1_total / ((idx0+1)*2)
        loss2_avg = loss2_total / ((idx0+1)*3)
        loss3_avg = loss3_total / ((idx0+1)*3)
        loss4_avg = loss4_total / ((idx0+1)*3)
        loss5_avg = loss5_total /((idx0+1)*3)

    print("eval total loss : %.9f " %(loss_avg))
    return predict_target_tensor , loss_avg, loss1_avg, loss2_avg, loss3_avg, loss4_avg, loss5_avg , attentionscores

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

def ComparePlot3Angles(truthAngle,PredAngle1,pred1name,PredAngle2,pred2name):
    fig2 = plt.figure(2)
    fig2.suptitle('Thumb finger angle prediction ',fontsize=20)
    ax1 = fig2.add_subplot(211)
    ax2 = fig2.add_subplot(212)
    #ax1.set_title('Thumb',fontsize=20)
    ax1.plot(truthAngle[0, :], color = 'k', label='truth')
    ax2.plot(truthAngle[1, :], color = 'k', label='truth')
    ax1.plot(PredAngle1[0, :], color = 'r',label=pred1name,alpha=0.5,linewidth = 0.7 )
    ax2.plot(PredAngle1[1, :], color = 'r',label=pred1name,alpha=0.5,linewidth = 0.7 )
    ax1.plot(PredAngle2[0, :], color='g', label=pred2name,alpha=0.9,linewidth = 0.7 )
    ax2.plot(PredAngle2[1, :], color='g', label=pred2name,alpha=0.9,linewidth = 0.7 )
    ax1.set_ylabel('rectified MCP angle',fontsize=10)
    ax1.set_xlabel('time (10ms)', fontsize=10)
    ax2.set_ylabel('rectified PIP angle',fontsize=10)
    ax2.set_xlabel('time (10ms)', fontsize=10)
    ax1.legend(loc='upper right')
    ax2.legend(loc='upper right')

    fig3 = plt.figure(3)
    fig3.suptitle('Index finger angle prediction', fontsize=20)
    ax1 = fig3.add_subplot(311)
    ax2 = fig3.add_subplot(312)
    ax3 = fig3.add_subplot(313)
    #ax1.set_title('index',fontsize=20)
    ax1.plot(truthAngle[2, :], color = 'k', label='truth')
    ax2.plot(truthAngle[3, :], color = 'k', label='truth')
    ax3.plot(truthAngle[4, :],color = 'k', label='truth')
    ax1.plot(PredAngle1[2, :], color = 'r',label=pred1name,alpha=0.3,linewidth = 0.7 )
    ax2.plot(PredAngle1[3, :], color = 'r',label=pred1name,alpha=0.3,linewidth = 0.7 )
    ax3.plot(PredAngle1[4, :],color = 'r',label=pred1name,alpha=0.3,linewidth = 0.7)
    ax1.plot(PredAngle2[2, :], color = 'g',label=pred2name,alpha=0.9,linewidth = 1)
    ax2.plot(PredAngle2[3, :], color = 'g',label=pred2name,alpha=0.9,linewidth = 1)
    ax3.plot(PredAngle2[4, :],color = 'g',label=pred2name,alpha=0.9,linewidth = 1)
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
    ax1.plot(PredAngle1[5, :],  color = 'r',label=pred1name)
    ax2.plot(PredAngle1[6, :],  color = 'r',label=pred1name)
    ax3.plot(PredAngle1[7, :],  color = 'r',label=pred1name)
    ax1.plot(PredAngle2[5, :],  color = 'g',label=pred2name)
    ax2.plot(PredAngle2[6, :],  color = 'g',label=pred2name)
    ax3.plot(PredAngle2[7, :],  color = 'g',label=pred2name)
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
    ax1.plot(PredAngle1[8, :], color = 'r',label=pred1name)
    ax2.plot(PredAngle1[9, :], color = 'r',label=pred1name)
    ax3.plot(PredAngle1[10, :], color = 'r',label=pred1name)
    ax1.plot(PredAngle2[8, :], color = 'g',label=pred2name)
    ax2.plot(PredAngle2[9, :], color = 'g',label=pred2name)
    ax3.plot(PredAngle2[10, :], color = 'g',label=pred2name)
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
    ax1.plot(PredAngle1[11, :], color = 'r',label=pred1name)
    ax2.plot(PredAngle1[12, :], color = 'r',label=pred1name)
    ax3.plot(PredAngle1[13, :], color = 'r',label=pred1name)
    ax1.plot(PredAngle2[11, :], color = 'g',label=pred2name)
    ax2.plot(PredAngle2[12, :], color = 'g',label=pred2name)
    ax3.plot(PredAngle2[13, :], color = 'g',label=pred2name)
    ax1.set_ylabel('MCP ',fontsize=20)
    ax2.set_ylabel('PIP',fontsize=20)
    ax3.set_ylabel('DIP', fontsize=20)
    ax1.legend(loc='upper right')
    ax2.legend(loc='upper right')
    ax3.legend(loc='upper right')

    plt.show()

def ComparePlot3Angles_inonefigrue(truthAngle,PredAngle1,pred1name,PredAngle2,pred2name):
    #truthAngle = truthAngle[:,2000:4000]
    #PredAngle1 = PredAngle1[:, 2000:4000]
    #PredAngle2 = PredAngle2[:, 2000:4000]

    color1 = 'g'
    alpha1 = 0.9
    color2 = 'r'

    plt.subplots_adjust(left=0.05, bottom=0.06, right=0.95, top=0.94, hspace=0.03, wspace=0)
    fig = plt.figure(1)
    plt.subplots_adjust(left=0.05, bottom=0.06, right=0.95, top=0.94, hspace=0.03, wspace=0)
    fig.suptitle('Finger angle prediction ',fontsize=20)
    ax_th1 = fig.add_subplot(14, 1, 1)
    ax_th2 = fig.add_subplot(14, 1, 2)
    ax_in1 = fig.add_subplot(14, 1, 3)
    ax_in2 = fig.add_subplot(14, 1, 4)
    ax_in3 = fig.add_subplot(14, 1, 5)
    ax_mi1 = fig.add_subplot(14, 1, 6)
    ax_mi2 = fig.add_subplot(14, 1, 7)
    ax_mi3 = fig.add_subplot(14, 1, 8)
    ax_ri1 = fig.add_subplot(14, 1, 9)
    ax_ri2 = fig.add_subplot(14, 1, 10)
    ax_ri3 = fig.add_subplot(14, 1, 11)
    ax_pi1 = fig.add_subplot(14, 1, 12)
    ax_pi2 = fig.add_subplot(14, 1, 13)
    ax_pi3 = fig.add_subplot(14, 1, 14)

    #ax1.set_title('Thumb',fontsize=20)
    ax_th1.plot(truthAngle[0, :], color = 'k', label='truth',linewidth = 1)
    ax_th2.plot(truthAngle[1, :], color = 'k', label='truth',linewidth = 1)
    ax_th1.plot(PredAngle1[0, :], color = color1,label=pred1name,alpha=0.9,linewidth = 1 )
    ax_th2.plot(PredAngle1[1, :], color = color1,label=pred1name,alpha=0.9,linewidth = 1 )
    ax_th1.plot(PredAngle2[0, :], color= color2, label=pred2name,alpha=0.9,linewidth = 1.3 )
    ax_th2.plot(PredAngle2[1, :], color= color2, label=pred2name,alpha=0.9,linewidth = 1.3 )
    ax_th1.set_ylabel('thumb\nMCP',fontsize=12)
    ax_th2.set_ylabel('thumb\nPIP',fontsize=12)
    ax_th1.legend(loc='upper right',prop={'size': 8})
    # ax_th2.legend(loc='upper right',prop={'size': 6})

    #ax1.set_title('index',fontsize=20)
    ax_in1.plot(truthAngle[2, :], color = 'k', label='truth',linewidth = 1)
    ax_in2.plot(truthAngle[3, :], color = 'k', label='truth',linewidth = 1)
    ax_in3.plot(truthAngle[4, :],color = 'k', label='truth',linewidth = 1)
    ax_in1.plot(PredAngle1[2, :], color = color1,label=pred1name,alpha=0.9,linewidth = 1)
    ax_in2.plot(PredAngle1[3, :], color = color1,label=pred1name,alpha=0.9,linewidth = 1)
    ax_in3.plot(PredAngle1[4, :],color = color1,label=pred1name,alpha=0.9,linewidth = 1)
    ax_in1.plot(PredAngle2[2, :], color = color2,label=pred2name,alpha=0.9,linewidth = 1.3)
    ax_in2.plot(PredAngle2[3, :], color = color2,label=pred2name,alpha=0.9,linewidth = 1.3)
    ax_in3.plot(PredAngle2[4, :],color = color2,label=pred2name,alpha=0.9,linewidth = 1.3)
    ax_in1.set_ylabel('Index\nMCP ',fontsize=12)
    ax_in2.set_ylabel('Index\nPIP',fontsize=12)
    ax_in3.set_ylabel('Index\nDIP', fontsize=12)
    # ax_in1.legend(loc='upper right')
    # ax_in2.legend(loc='upper right')
    # ax_in3.legend(loc='upper right')


    #ax1.title.set_text('middle',fontsize=20)
    ax_mi1.plot(truthAngle[5, :], color = 'k', label='truth',linewidth = 1)
    ax_mi2.plot(truthAngle[6, :], color = 'k', label='truth',linewidth = 1)
    ax_mi3.plot(truthAngle[7, :], color = 'k', label='truth',linewidth = 1)
    ax_mi1.plot(PredAngle1[5, :],  color = color1,label=pred1name,alpha=0.9,linewidth = 1)
    ax_mi2.plot(PredAngle1[6, :],  color = color1,label=pred1name,alpha=0.9,linewidth = 1)
    ax_mi3.plot(PredAngle1[7, :],  color = color1,label=pred1name,alpha=0.9,linewidth = 1)
    ax_mi1.plot(PredAngle2[5, :],  color = color2,label=pred2name,alpha=0.9,linewidth = 1.3)
    ax_mi2.plot(PredAngle2[6, :],  color = color2,label=pred2name,alpha=0.9,linewidth = 1.3)
    ax_mi3.plot(PredAngle2[7, :],  color = color2,label=pred2name,alpha=0.9,linewidth = 1.3)
    ax_mi1.set_ylabel('Middle\nMCP ',fontsize=12)
    ax_mi2.set_ylabel('Middle\nPIP',fontsize=12)
    ax_mi3.set_ylabel('Middle\nDIP', fontsize=12)
    # ax_mi1.legend(loc='upper right')
    # ax_mi2.legend(loc='upper right')
    # ax_mi3.legend(loc='upper right')


    #ax1.title.set_text('ring',fontsize=20)
    ax_ri1.plot(truthAngle[8, :], color = 'k', label='truth',linewidth = 1)
    ax_ri2.plot(truthAngle[9, :], color = 'k', label='truth',linewidth = 1)
    ax_ri3.plot(truthAngle[10, :], color = 'k', label='truth',linewidth = 1)
    ax_ri1.plot(PredAngle1[8, :], color = color1,label=pred1name,alpha=0.9,linewidth = 1)
    ax_ri2.plot(PredAngle1[9, :], color = color1,label=pred1name,alpha=0.9,linewidth = 1)
    ax_ri3.plot(PredAngle1[10, :], color = color1,label=pred1name,alpha=0.9,linewidth = 1)
    ax_ri1.plot(PredAngle2[8, :], color = color2,label=pred2name,alpha=0.9,linewidth = 1.3)
    ax_ri2.plot(PredAngle2[9, :], color = color2,label=pred2name,alpha=0.9,linewidth = 1.3)
    ax_ri3.plot(PredAngle2[10, :], color = color2,label=pred2name,alpha=0.9,linewidth = 1.3)
    ax_ri1.set_ylabel('Ring\nMCP ',fontsize=12)
    ax_ri2.set_ylabel('Ring\nPIP',fontsize=12)
    ax_ri3.set_ylabel('Ring\nDIP', fontsize=12)
    # ax_ri1.legend(loc='upper right')
    # ax_ri2.legend(loc='upper right')
    # ax_ri3.legend(loc='upper right')

    #ax1.title.set_text('pinky',fontsize=20)
    ax_pi1.plot(truthAngle[11, :], color = 'k', label='truth',linewidth = 1)
    ax_pi2.plot(truthAngle[12, :], color = 'k', label='truth',linewidth = 1)
    ax_pi3.plot(truthAngle[13, :], color = 'k', label='truth',linewidth = 1)
    ax_pi1.plot(PredAngle1[11, :], color = color1,label=pred1name,alpha=0.9,linewidth = 1)
    ax_pi2.plot(PredAngle1[12, :], color = color1,label=pred1name,alpha=0.9,linewidth = 1)
    ax_pi3.plot(PredAngle1[13, :], color = color1,label=pred1name,alpha=0.9,linewidth = 1)
    ax_pi1.plot(PredAngle2[11, :], color = color2,label=pred2name,alpha=0.9,linewidth = 1.3)
    ax_pi2.plot(PredAngle2[12, :], color = color2,label=pred2name,alpha=0.9,linewidth = 1.3)
    ax_pi3.plot(PredAngle2[13, :], color = color2,label=pred2name,alpha=0.9,linewidth = 1.3)
    ax_pi1.set_ylabel('Pinky\nMCP ',fontsize=12)
    ax_pi2.set_ylabel('Pinky\nPIP',fontsize=12)
    ax_pi3.set_ylabel('Pinky\nDIP', fontsize=12)
    ax_pi3.set_xlabel('time [10ms]', fontsize=15)
    # ax_pi1.legend(loc='upper right')
    # ax_pi2.legend(loc='upper right')
    # ax_pi3.legend(loc='upper right')
    plt.show()

def ComparePlot3Angles_inonefigrue_getfigure(truthAngle,PredAngle1,pred1name,PredAngle2,pred2name,path):
    #truthAngle = truthAngle[:,2000:4000]
    #PredAngle1 = PredAngle1[:, 2000:4000]
    #PredAngle2 = PredAngle2[:, 2000:4000]

    color1 = 'g'
    alpha1 = 0.9
    color2 = 'r'

    for i in tqdm(range(truthAngle.shape[1])) :
        #if i % 1 == 1 :
        plt.subplots_adjust(left=0.05, bottom=0.06, right=0.95, top=0.94, hspace=0.03, wspace=0)
        fig = plt.figure(1)
        plt.subplots_adjust(left=0.05, bottom=0.06, right=0.95, top=0.94, hspace=0.03, wspace=0)
        # fig.suptitle('Finger angle prediction ',fontsize=20)
        ax_th1 = fig.add_subplot(14, 1, 1)
        ax_th1.set_xlim([0, truthAngle.shape[1]])
        ax_th1.set_ylim([min(truthAngle[0, :]), max(truthAngle[0, :])])
        ax_th1.get_yaxis().set_visible(False)
        ax_th1.get_xaxis().set_visible(False)

        ax_th2 = fig.add_subplot(14, 1, 2)
        ax_th2.get_yaxis().set_visible(False)
        ax_th2.set_xlim([0, truthAngle.shape[1]])
        ax_th2.set_ylim([min(truthAngle[1, :]), max(truthAngle[1, :])])

        ax_in1 = fig.add_subplot(14, 1, 3)
        ax_in1.get_yaxis().set_visible(False)
        ax_in1.set_xlim([0, truthAngle.shape[1]])
        ax_in1.set_ylim([min(truthAngle[2, :]), max(truthAngle[2, :])])

        ax_in2 = fig.add_subplot(14, 1, 4)
        ax_in2.get_yaxis().set_visible(False)
        ax_in2.set_xlim([0, truthAngle.shape[1]])
        ax_in2.set_ylim([min(truthAngle[3, :]), max(truthAngle[3, :])])

        ax_in3 = fig.add_subplot(14, 1, 5)
        ax_in3.get_yaxis().set_visible(False)
        ax_in3.set_xlim([0, truthAngle.shape[1]])
        ax_in3.set_ylim([min(truthAngle[4, :]), max(truthAngle[4, :])])

        ax_mi1 = fig.add_subplot(14, 1, 6)
        ax_mi1.get_yaxis().set_visible(False)
        ax_mi1.set_xlim([0, truthAngle.shape[1]])
        ax_mi1.set_ylim([min(truthAngle[5, :]), max(truthAngle[5, :])])

        ax_mi2 = fig.add_subplot(14, 1, 7)
        ax_mi2.get_yaxis().set_visible(False)
        ax_mi2.set_xlim([0, truthAngle.shape[1]])
        ax_mi2.set_ylim([min(truthAngle[6, :]), max(truthAngle[6, :])])

        ax_mi3 = fig.add_subplot(14, 1, 8)
        ax_mi3.get_yaxis().set_visible(False)
        ax_mi3.set_xlim([0, truthAngle.shape[1]])
        ax_mi3.set_ylim([min(truthAngle[7, :]), max(truthAngle[7, :])])

        ax_ri1 = fig.add_subplot(14, 1, 9)
        ax_ri1.get_yaxis().set_visible(False)
        ax_ri1.set_xlim([0, truthAngle.shape[1]])
        ax_ri1.set_ylim([min(truthAngle[8, :]), max(truthAngle[8, :])])

        ax_ri2 = fig.add_subplot(14, 1, 10)
        ax_ri2.get_yaxis().set_visible(False)
        ax_ri2.set_xlim([0, truthAngle.shape[1]])
        ax_ri2.set_ylim([min(truthAngle[9, :]), max(truthAngle[9, :])])

        ax_ri3 = fig.add_subplot(14, 1, 11)
        ax_ri3.get_yaxis().set_visible(False)
        ax_ri3.set_xlim([0, truthAngle.shape[1]])
        ax_ri3.set_ylim([min(truthAngle[10, :]), max(truthAngle[10, :])])

        ax_pi1 = fig.add_subplot(14, 1, 12)
        ax_pi1.get_yaxis().set_visible(False)
        ax_pi1.set_xlim([0, truthAngle.shape[1]])
        ax_pi1.set_ylim([min(truthAngle[11, :]), max(truthAngle[11, :])])

        ax_pi2 = fig.add_subplot(14, 1, 13)
        ax_pi2.get_yaxis().set_visible(False)
        ax_pi2.set_xlim([0, truthAngle.shape[1]])
        ax_pi2.set_ylim([min(truthAngle[12, :]), max(truthAngle[12, :])])

        ax_pi3 = fig.add_subplot(14, 1, 14)
        ax_pi3.get_yaxis().set_visible(False)
        ax_pi3.get_xaxis().set_visible(False)
        ax_pi3.set_xlim([0, truthAngle.shape[1]])
        ax_pi3.set_ylim([min(truthAngle[13, :]), max(truthAngle[13, :])])

        #ax1.set_title('Thumb',fontsize=20)
        ax_th1.plot(truthAngle[0, :i], color = 'k', label='truth',linewidth = 1)
        ax_th2.plot(truthAngle[1, :i], color = 'k', label='truth',linewidth = 1)
        ax_th1.plot(PredAngle1[0, :i], color = color1,label=pred1name,alpha=0.9,linewidth = 1 )
        ax_th2.plot(PredAngle1[1, :i], color = color1,label=pred1name,alpha=0.9,linewidth = 1 )
        ax_th1.plot(PredAngle2[0, :i], color= color2, label=pred2name,alpha=0.9,linewidth = 1.3 )
        ax_th2.plot(PredAngle2[1, :i], color= color2, label=pred2name,alpha=0.9,linewidth = 1.3 )
        ax_th1.set_ylabel('thumb\nMCP',fontsize=12)
        ax_th2.set_ylabel('thumb\nPIP',fontsize=12)
        ax_th1.legend(loc='upper right',prop={'size': 8})
        # ax_th2.legend(loc='upper right',prop={'size': 6})

        #ax1.set_title('index',fontsize=20)
        ax_in1.plot(truthAngle[2, :i], color = 'k', label='truth',linewidth = 1)
        ax_in2.plot(truthAngle[3, :i], color = 'k', label='truth',linewidth = 1)
        ax_in3.plot(truthAngle[4, :i],color = 'k', label='truth',linewidth = 1)
        ax_in1.plot(PredAngle1[2, :i], color = color1,label=pred1name,alpha=0.9,linewidth = 1)
        ax_in2.plot(PredAngle1[3, :i], color = color1,label=pred1name,alpha=0.9,linewidth = 1)
        ax_in3.plot(PredAngle1[4, :i],color = color1,label=pred1name,alpha=0.9,linewidth = 1)
        ax_in1.plot(PredAngle2[2, :i], color = color2,label=pred2name,alpha=0.9,linewidth = 1.3)
        ax_in2.plot(PredAngle2[3, :i], color = color2,label=pred2name,alpha=0.9,linewidth = 1.3)
        ax_in3.plot(PredAngle2[4, :i],color = color2,label=pred2name,alpha=0.9,linewidth = 1.3)
        ax_in1.set_ylabel('Index\nMCP ',fontsize=12)
        ax_in2.set_ylabel('Index\nPIP',fontsize=12)
        ax_in3.set_ylabel('Index\nDIP', fontsize=12)
        # ax_in1.legend(loc='upper right')
        # ax_in2.legend(loc='upper right')
        # ax_in3.legend(loc='upper right')


        #ax1.title.set_text('middle',fontsize=20)
        ax_mi1.plot(truthAngle[5, :i], color = 'k', label='truth',linewidth = 1)
        ax_mi2.plot(truthAngle[6, :i], color = 'k', label='truth',linewidth = 1)
        ax_mi3.plot(truthAngle[7, :i], color = 'k', label='truth',linewidth = 1)
        ax_mi1.plot(PredAngle1[5, :i],  color = color1,label=pred1name,alpha=0.9,linewidth = 1)
        ax_mi2.plot(PredAngle1[6, :i],  color = color1,label=pred1name,alpha=0.9,linewidth = 1)
        ax_mi3.plot(PredAngle1[7, :i],  color = color1,label=pred1name,alpha=0.9,linewidth = 1)
        ax_mi1.plot(PredAngle2[5, :i],  color = color2,label=pred2name,alpha=0.9,linewidth = 1.3)
        ax_mi2.plot(PredAngle2[6, :i],  color = color2,label=pred2name,alpha=0.9,linewidth = 1.3)
        ax_mi3.plot(PredAngle2[7, :i],  color = color2,label=pred2name,alpha=0.9,linewidth = 1.3)
        ax_mi1.set_ylabel('Middle\nMCP ',fontsize=12)
        ax_mi2.set_ylabel('Middle\nPIP',fontsize=12)
        ax_mi3.set_ylabel('Middle\nDIP', fontsize=12)
        # ax_mi1.legend(loc='upper right')
        # ax_mi2.legend(loc='upper right')
        # ax_mi3.legend(loc='upper right')


        #ax1.title.set_text('ring',fontsize=20)
        ax_ri1.plot(truthAngle[8, :i], color = 'k', label='truth',linewidth = 1)
        ax_ri2.plot(truthAngle[9, :i], color = 'k', label='truth',linewidth = 1)
        ax_ri3.plot(truthAngle[10, :i], color = 'k', label='truth',linewidth = 1)
        ax_ri1.plot(PredAngle1[8, :i], color = color1,label=pred1name,alpha=0.9,linewidth = 1)
        ax_ri2.plot(PredAngle1[9, :i], color = color1,label=pred1name,alpha=0.9,linewidth = 1)
        ax_ri3.plot(PredAngle1[10, :i], color = color1,label=pred1name,alpha=0.9,linewidth = 1)
        ax_ri1.plot(PredAngle2[8, :i], color = color2,label=pred2name,alpha=0.9,linewidth = 1.3)
        ax_ri2.plot(PredAngle2[9, :i], color = color2,label=pred2name,alpha=0.9,linewidth = 1.3)
        ax_ri3.plot(PredAngle2[10, :i], color = color2,label=pred2name,alpha=0.9,linewidth = 1.3)
        ax_ri1.set_ylabel('Ring\nMCP ',fontsize=12)
        ax_ri2.set_ylabel('Ring\nPIP',fontsize=12)
        ax_ri3.set_ylabel('Ring\nDIP', fontsize=12)
        # ax_ri1.legend(loc='upper right')
        # ax_ri2.legend(loc='upper right')
        # ax_ri3.legend(loc='upper right')

        #ax1.title.set_text('pinky',fontsize=20)
        ax_pi1.plot(truthAngle[11, :i], color = 'k', label='truth',linewidth = 1)
        ax_pi2.plot(truthAngle[12, :i], color = 'k', label='truth',linewidth = 1)
        ax_pi3.plot(truthAngle[13, :i], color = 'k', label='truth',linewidth = 1)
        ax_pi1.plot(PredAngle1[11, :i], color = color1,label=pred1name,alpha=0.9,linewidth = 1)
        ax_pi2.plot(PredAngle1[12, :i], color = color1,label=pred1name,alpha=0.9,linewidth = 1)
        ax_pi3.plot(PredAngle1[13, :i], color = color1,label=pred1name,alpha=0.9,linewidth = 1)
        ax_pi1.plot(PredAngle2[11, :i], color = color2,label=pred2name,alpha=0.9,linewidth = 1.3)
        ax_pi2.plot(PredAngle2[12, :i], color = color2,label=pred2name,alpha=0.9,linewidth = 1.3)
        ax_pi3.plot(PredAngle2[13, :i], color = color2,label=pred2name,alpha=0.9,linewidth = 1.3)
        ax_pi1.set_ylabel('Pinky\nMCP ',fontsize=12)
        ax_pi2.set_ylabel('Pinky\nPIP',fontsize=12)
        ax_pi3.set_ylabel('Pinky\nDIP', fontsize=12)
        #ax_pi3.set_xlabel('time [10ms]', fontsize=15)
        # ax_pi1.legend(loc='upper right')
        # ax_pi2.legend(loc='upper right')
        # ax_pi3.legend(loc='upper right')
        # plt.show()

        savename = path + '/' + "{:09n}".format(i) + '.png'
        plt.savefig(savename)
        plt.cla()
        plt.clf()
        plt.close()





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
    save_path = '/home/hyuninlee/PycharmProjects/xcorps/seq2seq_attentionmodel/result/Attention_new/'
    model_path = '/home/hyuninlee/PycharmProjects/xcorps/seq2seq_attentionmodel/modeldata/'
    N_emgsensor = 4
    N_fingerAngle = 14

    torch.manual_seed(0)


    #variable hyperparameters
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--hidden_size', type=int, default=256)
    parser.add_argument('--teacher_forcing_ratio','-tfr' ,type=float, default=0.5)
    parser.add_argument('--train_iter','-ti' ,type=int, default=100)
    parser.add_argument('--learning_rate_encoder','-lre' ,type=float, default=0.0004)
    parser.add_argument('--learning_rate_decoder', '-lrd', type=float, default=0.0005)
    parser.add_argument('--time_length','-tl' ,type=int, default=1)
    parser.add_argument('--batch_size', type=int, default=1024)
    parser.add_argument('--doesTest', type=bool, default=True)
    parser.add_argument('--doesEval', type=bool, default=True)
    args = parser.parse_args()

    params = vars(args) #convert args to dirctionary

    hidden_size = params['hidden_size']
    teacher_forcing_ratio = params['teacher_forcing_ratio']
    train_iter = params['train_iter']
    learning_rate_encoder = params['learning_rate_encoder']
    learning_rate_decoder = params['learning_rate_decoder']
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
    name = 'lre_' + str(learning_rate_encoder)+'_lrd_' + str(learning_rate_decoder)+'_bs_'+str(batch_size)+'_tfr_'+str(teacher_forcing_ratio)+'_tl_'+str(time_length)+'__'+time.strftime("%d-%m-%Y_%H-%M-%S")

    writer = SummaryWriter(log_dir = save_path+name)

    #prpare data
    input_data, output_data,input_data_eval,output_data_eval = dataprepare(data_path,doesEval)

    #convert output_data and output_data_eval 1 <-> 0
    output_data = 1-output_data
    output_data_eval = 1-output_data_eval

    #model
    encoder1 = EncoderRNN(N_emgsensor, hidden_size).to(device)
    attn_decoder_thumb = AttnDecoderRNN(hidden_size, 2, dropout_p=0.1,decoder_time_length=N_emgsensor).to(device)
    attn_decoder_index = AttnDecoderRNN(hidden_size, 3, dropout_p=0.1, decoder_time_length=N_emgsensor).to(device)
    attn_decoder_middle = AttnDecoderRNN(hidden_size, 3, dropout_p=0.1, decoder_time_length=N_emgsensor).to(device)
    attn_decoder_ring = AttnDecoderRNN(hidden_size, 3, dropout_p=0.1, decoder_time_length=N_emgsensor).to(device)
    attn_decoder_pinky = AttnDecoderRNN(hidden_size, 3, dropout_p=0.1, decoder_time_length=N_emgsensor).to(device)

    #summary(encoder1 , [(batch_size, N_emgsensor),(batch_size, hidden_size)])

    # print parameter nyumber of encoder1 and attn_decoder1
    model_parameters = filter(lambda p: p.requires_grad, encoder1.parameters())
    params = sum([np.prod(p.size()) for p in model_parameters])
    print(params)
    model_parameters = filter(lambda p: p.requires_grad, attn_decoder_thumb.parameters())
    params = sum([np.prod(p.size()) for p in model_parameters])
    print(params)
    model_parameters = filter(lambda p: p.requires_grad, attn_decoder_index.parameters())
    params = sum([np.prod(p.size()) for p in model_parameters])
    print(params)
    model_parameters = filter(lambda p: p.requires_grad, attn_decoder_middle.parameters())
    params = sum([np.prod(p.size()) for p in model_parameters])
    print(params)
    model_parameters = filter(lambda p: p.requires_grad, attn_decoder_ring.parameters())
    params = sum([np.prod(p.size()) for p in model_parameters])
    print(params)
    model_parameters = filter(lambda p: p.requires_grad, attn_decoder_pinky.parameters())
    params = sum([np.prod(p.size()) for p in model_parameters])
    print(params)

    #train
    trainIters(input_data, output_data,input_data_eval,output_data_eval,time_length, encoder1, attn_decoder_thumb,
               attn_decoder_index,attn_decoder_middle,attn_decoder_ring,attn_decoder_pinky,train_iter, eval_every=1, test_every=1,
               learning_rate_encoder=learning_rate_encoder, learning_rate_decoder=learning_rate_decoder,batch_size = batch_size)


    writer.flush()
    writer.close()

    #prepare sequential evaluation set.

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


