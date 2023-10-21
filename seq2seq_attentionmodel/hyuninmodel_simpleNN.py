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

#### set random_seed_number ###
random_seed_number = 1



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

class simpleNN(nn.Module) :
    def __init__(self,input,output,hidden):
        super(simpleNN, self).__init__()
        self.layer1 = nn.Linear(input,hidden)
        self.leakyrelu = nn.LeakyReLU(0.1)
        self.layer2 = nn.Linear(hidden,hidden * 2)
        self.layer3 = nn.Linear(hidden * 2, hidden * 2)
        self.layer4 = nn.Linear(hidden * 2, hidden)
        self.layer5 = nn.Linear(hidden,output)
        # self.drop1 = nn.Dropout(p=0.2)
        # self.drop2 = nn.Dropout(p=0.4)

    def forward(self,inputData):
        x = self.layer2(self.layer1(inputData))
        x = self.leakyrelu(x)
        x = self.layer3(x)
        x = self.leakyrelu(x)
        x = self.layer4(x)
        x = self.leakyrelu(x)
        outputData = self.layer5(x)

        return outputData

def train(input_tensor, target_tensor, time_length, simplenetwork, simplenetwork_optimizer, criterion,tfr_prob_list,iter):
    #input_tensor : (time_length,1,bs,4)
    #target_tensor : (time_length,1,bs,14)

    assert input_tensor.shape[0] == 1
    input_tensor = input_tensor.squeeze(dim=0).squeeze(dim=0)
    target_tensor = target_tensor.squeeze(dim=0).squeeze(dim=0)

    output = simplenetwork(input_tensor)

    simplenetwork_optimizer.zero_grad()
    loss = 0

    loss += criterion(output, target_tensor)

    loss.backward()

    simplenetwork_optimizer.step()

    return loss.item() / time_length

def trainIters(input_data,output_data,input_data_eval,output_data_eval,time_length, simplenetwork , n_epochs, print_every, plot_every, learning_rate,batch_size):
    #input_data : (4,data_length)
    #output_data  : (14,data_length)

    #prepare test data
    from scipy.signal import savgol_filter
    test_input_data, test_output_data,_,_ = dataprepare(test_path, test=True)

    # shift dataset
    test_input_data = test_input_data[:, :test_input_data.shape[1] - shiftLength]
    new_test_output_data = np.zeros((14, test_input_data.shape[1]))
    for idx in range(new_test_output_data.shape[1]):
        new_test_output_data[:, idx] = test_output_data[:, idx + shiftLength]


    test_output_data = new_test_output_data
    del new_test_output_data
    # test_output_data convert 0 <-> 1
    test_output_data = 1-test_output_data


    start = time.time()
    print_loss_total = 0  # Reset every print_every

    simplenetwork_optimizer = optim.Adam(simplenetwork.parameters(), lr=learning_rate)
    criterion = nn.MSELoss()


    for epoch in range(1, n_epochs + 1):
        print('========== epoch : %d =========='% (epoch))
        ###################################
        simplenetwork.train()
        #########################
        randomindex = [x for x in range(input_data.shape[1]-time_length)]
        random.Random(epoch+random_seed_number).shuffle(randomindex)
        num_iters = (input_data.shape[1]-time_length)//batch_size

        tfr_prob_list = np.random.random(num_iters)

        for iter in range(num_iters):
            input_tensor, target_tensor = dataloader(iter, time_length, input_data, output_data, randomindex,batchsize=batch_size)

            #np.random.seed(epoch+random_seed_number)
            loss = train(input_tensor, target_tensor, time_length, simplenetwork, simplenetwork_optimizer, criterion, tfr_prob_list,iter)
            writer.add_scalar('Loss/iter',loss,(epoch-1)*num_iters + iter)
            print_loss_total += loss

            if iter % int(0.3*(input_data.shape[1]-time_length)//batch_size) == 0 :
                print('iter : %d , loss : %.9f' % (iter, loss))
        print_loss_avg = print_loss_total/num_iters
        writer.add_scalar('Loss/epoch', print_loss_avg, epoch)
        print_loss_total = 0

        ###################################
        simplenetwork.eval()
        #########################

        if epoch % print_every == 0:
            print('%s (%d %d%%) loss_avg : %.9f' % (timeSince(start, epoch / n_epochs),
                                         epoch, epoch / n_epochs * 100, print_loss_avg))
            _, eval_loss_avg = test(input_data_eval, output_data_eval, time_length, simplenetwork)
            _, test_loss_avg = test(test_input_data, test_output_data, time_length, simplenetwork)
            writer.add_scalar('Loss/eval', eval_loss_avg, epoch)
            writer.add_scalar('Loss/test', test_loss_avg, epoch)

        print('==============================')
    #showPlot(plot_losses)


def test(input_data, output_data, time_length, simplenetwork):

    loss_list = []
    criterion = nn.MSELoss()
    loss = 0
    input_tensor_list, target_tensor_list = testdataloader(time_length,input_data,output_data)
    predict_target_tensor = np.zeros_like(output_data)

    with torch.no_grad() :
        for idx,(input_tensor , target_tensor) in enumerate(zip(input_tensor_list,target_tensor_list)):
            assert input_tensor.shape[0] == 1
            input_tensor = input_tensor.squeeze(dim=0).squeeze(dim=0)
            target_tensor = target_tensor.squeeze(dim=0).squeeze(dim=0)
            output = torch.transpose(simplenetwork(input_tensor),0,1)
            predict_target_tensor[:,idx] = output.cpu().numpy().squeeze()

            #a = np.transpose(predict_target_tensor[:,idx]).astype(np.float32)
            #a = np.expand_dims(a,axis=1)
            #b = torch.transpose(target_tensor, 0, 1)
            #b = b.cpu().numpy()
            loss += criterion( output ,torch.transpose(target_tensor, 0, 1))

        loss = loss/(idx+1)
        assert (idx+1) == len(input_tensor_list)

            #loss_list.append(loss.item()/time_length)
    #loss_avg = sum(loss_list)/len(loss_list)
    #print("eval loss : %.9f " %(loss_avg))
    return predict_target_tensor , loss

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
    ax1 = fig2.add_subplot(211)
    ax2 = fig2.add_subplot(212)
    ax1.title.set_text('Thumb')
    ax1.plot(truthAngle[0, :], label='0')
    ax2.plot(truthAngle[1, :], label='1')
    ax1.plot(PredAngle[0, :], label='0')
    ax2.plot(PredAngle[1, :], label='1')

    fig3 = plt.figure(3)
    ax1 = fig3.add_subplot(311)
    ax2 = fig3.add_subplot(312)
    ax3 = fig3.add_subplot(313)
    ax1.title.set_text('index')
    ax1.plot(truthAngle[2, :], label='0')
    ax2.plot(truthAngle[3, :], label='1')
    ax3.plot(truthAngle[4, :], label='1')
    ax1.plot(PredAngle[2, :], label='0')
    ax2.plot(PredAngle[3, :], label='1')
    ax3.plot(PredAngle[4, :], label='1')

    fig4 = plt.figure(4)
    ax1 = fig4.add_subplot(311)
    ax2 = fig4.add_subplot(312)
    ax3 = fig4.add_subplot(313)
    ax1.title.set_text('middle')
    ax1.plot(truthAngle[5, :], label='0')
    ax2.plot(truthAngle[6, :], label='1')
    ax3.plot(truthAngle[7, :], label='1')
    ax1.plot(PredAngle[5, :], label='0')
    ax2.plot(PredAngle[6, :], label='1')
    ax3.plot(PredAngle[7, :], label='1')

    fig5 = plt.figure(5)
    ax1 = fig5.add_subplot(311)
    ax2 = fig5.add_subplot(312)
    ax3 = fig5.add_subplot(313)
    ax1.title.set_text('ring')
    ax1.plot(truthAngle[8, :], label='0')
    ax2.plot(truthAngle[9, :], label='1')
    ax3.plot(truthAngle[10, :], label='1')
    ax1.plot(PredAngle[8, :], label='0')
    ax2.plot(PredAngle[9, :], label='1')
    ax3.plot(PredAngle[10, :], label='1')

    fig6 = plt.figure(6)
    ax1 = fig6.add_subplot(311)
    ax2 = fig6.add_subplot(312)
    ax3 = fig6.add_subplot(313)
    ax1.title.set_text('pinky')
    ax1.plot(truthAngle[11, :], label='0')
    ax2.plot(truthAngle[12, :], label='1')
    ax3.plot(truthAngle[13, :], label='1')
    ax1.plot(PredAngle[11, :], label='0')
    ax2.plot(PredAngle[12, :], label='1')
    ax3.plot(PredAngle[13, :], label='1')

    plt.show()


if __name__=="__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # fixed hyperparameters
    data_path = '/home/hyuninlee/PycharmProjects/xcorps/seq2seq_attentionmodel/trainData/lhi'
    test_path = '/home/hyuninlee/PycharmProjects/xcorps/seq2seq_attentionmodel/testData/lhi'
    save_path = '/home/hyuninlee/PycharmProjects/xcorps/seq2seq_attentionmodel/result/simpleNN/'
    model_path = '/home/hyuninlee/PycharmProjects/xcorps/seq2seq_attentionmodel/modeldata/'
    N_emgsensor = 4
    N_fingerAngle = 14

    torch.manual_seed(random_seed_number)


    #variable hyperparameters
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--shift_length','-sl', type=int, default=0)
    parser.add_argument('--hidden_size', type=int, default=256)
    parser.add_argument('--teacher_forcing_ratio','-tfr' ,type=float, default=0.5)
    parser.add_argument('--train_iter', type=int, default=100)
    parser.add_argument('--learning_rate','-lr' ,type=float, default=0.003)
    parser.add_argument('--time_length', type=int, default=1)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--doesTest', type=bool, default=True)
    parser.add_argument('--doesEval', type=bool, default=True)
    args = parser.parse_args()

    params = vars(args) #convert args to dirctionary

    shiftLength = params['shift_length']
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

    name = 'sl_'+str(shiftLength)+'_lr_' + str(learning_rate)+'_bs_'+str(batch_size)+'_tfr_'+str(teacher_forcing_ratio)+'__'+time.strftime("%d-%m-%Y_%H-%M-%S")

    writer = SummaryWriter(log_dir = save_path+name)

    #prpare data
    input_data, output_data,input_data_eval,output_data_eval = dataprepare(data_path,doesEval)



    #shift dataset
    # input_data = input_data[:,:input_data.shape[1]-shiftLength]
    # input_data_eval = input_data_eval[:,:input_data_eval.shape[1]-shiftLength]
    # new_output_data = np.zeros((14,input_data.shape[1]))
    # new_output_data_eval = np.zeros((14,input_data_eval.shape[1]))
    # for idx in range(new_output_data.shape[1])  :
    #     new_output_data[:,idx] = output_data[:,idx+shiftLength]
    # for idx in range(new_output_data_eval.shape[1]):
    #     new_output_data_eval[:,idx] = output_data_eval[:,idx+shiftLength]
    #
    # output_data = new_output_data
    # output_data_eval = new_output_data_eval
    # del new_output_data
    # del new_output_data_eval



    #convert output_data and output_data_eval 1 <-> 0
    output_data = 1-output_data
    output_data_eval = 1-output_data_eval

    #model
    # encoder1 = EncoderRNN(N_emgsensor, hidden_size).to(device)
    # attn_decoder1 = AttnDecoderRNN(hidden_size, N_fingerAngle, dropout_p=0.1,time_length=time_length,).to(device)

    SimpleNetwork = simpleNN(N_emgsensor,N_fingerAngle,hidden_size).to(device)
    model_parameters = filter(lambda p: p.requires_grad, SimpleNetwork.parameters())
    params = sum([np.prod(p.size()) for p in model_parameters])
    print(params)

    #train
    train_param = {
        "innput_data" : input_data,
        "output_data" : output_data,
        "time_length" : time_length,
        "sinpleNN" : SimpleNetwork,
        "n_epochs" : train_iter,
        "print_every" : 1,
        "plot_every" : 1,
        "learning_rate" : learning_rate,
        "batchsize" : batch_size,
    }
    trainIters(input_data, output_data,input_data_eval,output_data_eval,time_length, SimpleNetwork, train_iter, print_every=1, plot_every=1, learning_rate=learning_rate,batch_size = batch_size)

    torch.save(SimpleNetwork.state_dict(), model_path+name)

    writer.flush()
    writer.close()

    # if doesTest :
    #     from scipy.signal import savgol_filter
    #     test_input_data, test_output_data,_,_ = dataprepare(test_path, test=True)
    #
    #     # shift dataset
    #     test_input_data = test_input_data[:, :test_input_data.shape[1] - shiftLength]
    #     new_test_output_data = np.zeros((14, test_input_data.shape[1]))
    #     for idx in range(new_test_output_data.shape[1]):
    #         new_test_output_data[:, idx] = test_output_data[:, idx + shiftLength]
    #
    #
    #     test_output_data = new_test_output_data
    #     del new_test_output_data
    #     # test_output_data convert 0 <-> 1
    #     test_output_data = 1-test_output_data
    #
    #     test_pred_target,_ = test(test_input_data,test_output_data,time_length, SimpleNetwork)
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


