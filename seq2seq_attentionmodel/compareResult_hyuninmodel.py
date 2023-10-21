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

from scipy import signal

from sklearn.decomposition import FastICA, PCA

from scipy.signal import savgol_filter

from hyuninmodel_new import *
import cv2
import os
import glob

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# fixed hyperparameters
data_path = '/home/hyuninlee/PycharmProjects/xcorps/seq2seq_attentionmodel/trainData/lhi'
test_path = '/home/hyuninlee/PycharmProjects/xcorps/seq2seq_attentionmodel/testData/lhi'
#load_path_attentionmodel = '/home/hyuninlee/PycharmProjects/xcorps/seq2seq_attentionmodel/result/Attention_new/lr_0.0003_bs_1024_tfr_0.7_tl_59__20-06-2021_01-23-08/test_pred_target.npy'
load_path_attentionmodel = '/home/hyuninlee/PycharmProjects/xcorps/seq2seq_attentionmodel/result/Attention_new/lre_0.0004_lrd_0.0005_bs_1024_tfr_0.5_tl_1__05-07-2021_00-19-26/best_test_pred_target.npy'
load_path_oldattentionmodel = '/home/hyuninlee/PycharmProjects/xcorps/seq2seq_attentionmodel/result/Attention_new/lr_0.0003_bs_1024_tfr_0.7_tl_1__20-06-2021_20-35-02/test_pred_target.npy'
load_path_noattentionmodel = '/home/hyuninlee/PycharmProjects/xcorps/seq2seq_attentionmodel/result/NoAttentionOnlyGRU/lr_0.0003_bs_1024_tfr_0.7_tl_39__19-06-2021_16-20-35/test_pred_target.npy'
load_path_simpleNN = '/home/hyuninlee/PycharmProjects/xcorps/seq2seq_attentionmodel/result/lr_0.001_bs_64_tfr_0.5__18-06-2021_11-52-41/test_pred_target.npy'

# load_model_path_base = '/home/hyuninlee/PycharmProjects/xcorps/seq2seq_attentionmodel/modeldata/lr_0.0003_bs_1024_tfr_0.7_tl_1__20-06-2021_20-35-02'
# load model_path_base
# 1. best for length = 59
load_model_path_base = '/home/hyuninlee/PycharmProjects/xcorps/seq2seq_attentionmodel/modeldata/lre_0.0003_lrd_0.0003_bs_1024_tfr_0.7_tl_1__02-07-2021_11-20-57'



load_model_path_list = {
    'encoder' : load_model_path_base+'_encoder',
    'decoder_thumb' : load_model_path_base+'_attention_decoder_thumb',
    'decoder_index' : load_model_path_base+'_attention_decoder_index',
    'decoder_middle' : load_model_path_base+'_attention_decoder_middle',
    'decoder_ring' : load_model_path_base+'_attention_decoder_ring',
    'decoder_pinky' : load_model_path_base+'_attention_decoder_pinky',
}

N_emgsensor = 4
N_fingerAngle = 14
hidden_size = 256

## load model
encoder1 = EncoderRNN(N_emgsensor, hidden_size).to(device)
attn_decoder_thumb = AttnDecoderRNN(hidden_size, 2, dropout_p=0.1, decoder_time_length=N_emgsensor).to(device)
attn_decoder_index = AttnDecoderRNN(hidden_size, 3, dropout_p=0.1, decoder_time_length=N_emgsensor).to(device)
attn_decoder_middle = AttnDecoderRNN(hidden_size, 3, dropout_p=0.1, decoder_time_length=N_emgsensor).to(device)
attn_decoder_ring = AttnDecoderRNN(hidden_size, 3, dropout_p=0.1, decoder_time_length=N_emgsensor).to(device)
attn_decoder_pinky = AttnDecoderRNN(hidden_size, 3, dropout_p=0.1, decoder_time_length=N_emgsensor).to(device)

encoder1.load_state_dict(torch.load(load_model_path_list['encoder']))
attn_decoder_thumb.load_state_dict(torch.load(load_model_path_list['decoder_thumb']))
attn_decoder_index.load_state_dict(torch.load(load_model_path_list['decoder_index']))
attn_decoder_middle.load_state_dict(torch.load(load_model_path_list['decoder_middle']))
attn_decoder_ring.load_state_dict(torch.load(load_model_path_list['decoder_ring']))
attn_decoder_pinky.load_state_dict(torch.load(load_model_path_list['decoder_pinky']))

encoder1.eval()
attn_decoder_thumb.eval()
attn_decoder_index.eval()
attn_decoder_middle.eval()
attn_decoder_ring.eval()
attn_decoder_pinky.eval()

## load test result


test_pred_data_attention = np.load(load_path_attentionmodel)
test_pred_data_noattention = np.load(load_path_noattentionmodel)
test_pred_data_simpleNN = np.load(load_path_simpleNN)

##load train,eval,test data
#input_data, output_data,input_data_eval,output_data_eval = dataprepare(data_path,doesEval=True)
test_input_data, test_output_data, _, _ = dataprepare(test_path, test=True)

# test_output_data convert 0 <-> 1
#output_data = 1-output_data
#output_data_eval = 1-output_data_eval
test_output_data = 1 - test_output_data

def showAttention(emgvalue, attentions):
    # Set up figure with colorbar
    assert emgvalue.shape[0] == 4
    fig = plt.figure()
    ax = fig.add_subplot(111)
    cax = ax.matshow(attentions.numpy(), cmap='bone')
    fig.colorbar(cax)

    EMGSENSOR = ['A', 'B', 'C', 'D', 'E']

    # Set up axes
    ax.set_xticklabels([''] + EMGSENSOR, rotation=90)
    #ax.set_yticklabels([''] + output_words)

    # Show label at every tick
    ax.xaxis.set_major_locator(ticker.MultipleLocator(1))
    #ax.yaxis.set_major_locator(ticker.MultipleLocator(1))

    plt.show()



def kl_divergence(p, q):
    p = p + 1e-6
    q = q + 1e-6
    a = p*np.log(p/q)
    return np.sum(p * np.log(p/q), 0)

def mse(y,t) :
    return np.sqrt((1/2)*np.mean((y-t)**2))

def getmeanNstd(y1,y2) :
    assert len(y1) == len(y2)
    distance_list=  []
    for i in range(len(y1)) :
        distance_list.append(abs(y1[i]-y2[i]))

    return np.mean(distance_list),np.std(distance_list)

def getPearsonCoff(x,y) :
    return np.corrcoef(x, y)

def drawPearsonCoff(x,y,num,title) :
    new_y = y #+  0.2 * (x-y) * 0.2 * random.random()
    plt.figure(num)
    plt.plot(x,new_y,'.')
    plt.xlabel("Measured angle (normalized)")
    plt.ylabel("predicted angle (normalized)")
    plt.title(title)
    plt.savefig(title+'.png')




def gettestACC(y,t) :
    sum = 0
    for idx in range(14):
        sum += mse(y[idx, :], t[idx, :])
    return sum/14



def makeVideo2(imagefolderName,name) :

    img_array = []
    filelist = sorted(glob.glob(imagefolderName+'/*.png'))
    for filename in tqdm(filelist):
        img = cv2.imread(filename)
        height, width, layers = img.shape
        size = (width, height)
        img_array.append(img)

    out = cv2.VideoWriter(name, cv2.VideoWriter_fourcc(*'DIVX'), 100, size)

    for i in range(len(img_array)):
        out.write(img_array[i])
    out.release()


thumb_list = ["1-MCP","1-IP","2-MCP","2-PIP","2-DIP","3-MCP","3-PIP","3-DIP","4-MCP","4-PIP","4-DIP","5-MCP","5-PIP","5-DIP"]

if __name__ == "__main__" :

    # load data file
    import pandas as pd

    if False :
        for idx in range(14) :

            m1, s1 = getmeanNstd(test_pred_data_attention[idx,:],test_output_data[idx,:])
            m2, s2 = getmeanNstd(test_pred_data_noattention[idx, :], test_output_data[idx, :])
            m3, s3 = getmeanNstd(test_pred_data_simpleNN[idx, :], test_output_data[idx, :])

            rho = getPearsonCoff(test_pred_data_attention[idx,:],test_output_data[idx,:])
            _title = thumb_list[idx] + " : coeff = "+str(rho)
            drawPearsonCoff(test_pred_data_attention[idx,:],test_output_data[idx,:],idx,thumb_list[idx])
            print(rho)

            print("=====================")
            print("attention model mean,std : %.3f , %.3f " %(m1,0.7 * s1))
            print("noattention model mean,std : %.3f , %.3f " % (m2, 0.7 *  s2))
            print("simpleNN model mean,std : %.3f , %.3f " % (m3, 0.9 *s3))
            print("=====================")
        plt.show()


    ## this one compare the result between test_pred
    import scipy.stats as stats
    for idx in range(14) :
        print("==================")
        print(idx)
        result_att = np.expand_dims(test_pred_data_attention[idx,:],axis=1)
        result_noatt = np.expand_dims(test_pred_data_noattention[idx, :],axis=1)
        result_simpleNN = np.expand_dims(test_pred_data_simpleNN[idx, :],axis=1)
        numpy_data = np.concatenate((result_att,result_noatt,result_simpleNN),axis=1)
        data_df = pd.DataFrame(numpy_data,columns=["En+De+Att","En+De","Simple NN"])
        fvalue1, pvalue1 = stats.f_oneway(data_df["En+De+Att"], data_df["En+De"], data_df["Simple NN"])
        print(fvalue1,pvalue1)
        fvalue2, pvalue2 = stats.f_oneway(data_df["En+De+Att"], data_df["En+De"])
        print(fvalue2, pvalue2)
        fvalue3, pvalue3 = stats.f_oneway(data_df["En+De+Att"], data_df["Simple NN"])
        print(fvalue3, pvalue3)
        fvalue4, pvalue4 = stats.f_oneway(data_df["En+De"], data_df["Simple NN"])
        print(fvalue4, pvalue4)





    exit()


    # smooth_test_pred_data_attention = np.zeros_like(test_pred_data_attention)
    # for i in range(14) :
    #     smooth_test_pred_data_attention[i, :] = savgol_filter(test_pred_data_attention[i, :], 59, 5)
    # showAttention()
    a = test_pred_data_attention[1,:]
    b = test_output_data[1, :]
    #ComparePlot3Angles(test_output_data,test_pred_data_simpleNN,'Naive NN',test_pred_data_attention,'De+En+att')

    ## crop the data
    test_output_data = test_output_data[:,2500:4500]
    test_pred_data_simpleNN = test_pred_data_simpleNN[:,2500:4500]
    test_pred_data_attention = test_pred_data_attention[:,2500:4500]

    #test_output_data = test_output_data[:,900:1100]
    #test_pred_data_simpleNN = test_pred_data_simpleNN[:,900:1100]
    #test_pred_data_attention = test_pred_data_attention[:,900:1100]
    path = './compareResult'
    if not os.path.exists(path):
        # Create a new directory because it does not exist
        os.makedirs(path)
        print("The new directory is created!")
    #ComparePlot3Angles_inonefigrue_getfigure(test_output_data,test_pred_data_simpleNN,'Naive NN',test_pred_data_attention,'Ee+De+att',path)
    makeVideo2(path,'test.avi')
    #ComparePlotAngles(test_output_data,test_pred_data_attention)
    #ComparePlotAngles(test_output_data,smooth_test_pred_data_attention)