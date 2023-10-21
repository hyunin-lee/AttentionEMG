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

from hyuninmodel import *

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# fixed hyperparameters
data_path = '/home/hyuninlee/PycharmProjects/xcorps/seq2seq_attentionmodel/trainData/lhi/2sec'
test_path = '/home/hyuninlee/PycharmProjects/xcorps/seq2seq_attentionmodel/testData/lhi'
#this is the best attentionmodel when length =29,39,49,59,69
load_model_path_base_19 = '/home/hyuninlee/PycharmProjects/xcorps/seq2seq_attentionmodel/result/Attention_old_2/lr_0.0005_bs_1024_tfr_0.8_tl_19__22-06-2021_12-15-17'
load_model_path_base_29 = '/home/hyuninlee/PycharmProjects/xcorps/seq2seq_attentionmodel/result/Attention_old_2/lr_0.0003_bs_1024_tfr_0.7_tl_29__22-06-2021_12-40-36'
load_model_path_base_39 = '/home/hyuninlee/PycharmProjects/xcorps/seq2seq_attentionmodel/result/Attention_old_2/lr_0.0007_bs_1024_tfr_0.7_tl_39__22-06-2021_14-20-31'
load_model_path_base_49 = '/home/hyuninlee/PycharmProjects/xcorps/seq2seq_attentionmodel/result/Attention_old_2/lr_0.0001_bs_1024_tfr_0.8_tl_49__22-06-2021_16-10-38'
load_model_path_base_59 = '/home/hyuninlee/PycharmProjects/xcorps/seq2seq_attentionmodel/result/Attention_old_2/lr_0.0003_bs_1024_tfr_0.7_tl_59__22-06-2021_16-53-01'
load_model_path_base_69 = '/home/hyuninlee/PycharmProjects/xcorps/seq2seq_attentionmodel/result/Attention_old_2/lr_0.0003_bs_1024_tfr_0.7_tl_69__22-06-2021_18-09-26'
load_model_path_base_79 = '/home/hyuninlee/PycharmProjects/xcorps/seq2seq_attentionmodel/result/Attention_old_2/lr_0.0005_bs_1024_tfr_0.7_tl_79__23-06-2021_00-07-53'
load_model_path_base_89 = '/home/hyuninlee/PycharmProjects/xcorps/seq2seq_attentionmodel/result/Attention_old_2/lr_0.0007_bs_1024_tfr_0.7_tl_89__23-06-2021_01-55-09'



#this is the best noattentionmodel when length=39
load_path_noattentionmodel = '/home/hyuninlee/PycharmProjects/xcorps/seq2seq_attentionmodel/result/NoAttentionOnlyGRU/lr_0.0003_bs_1024_tfr_0.7_tl_39__19-06-2021_16-20-35/test_pred_target.npy'
#this is the nest simpleNN model
load_path_simpleNN = '/home/hyuninlee/PycharmProjects/xcorps/seq2seq_attentionmodel/result/lr_0.001_bs_64_tfr_0.5__18-06-2021_11-52-41/test_pred_target.npy'
#load oldattentionmodel


# load_model_path_list = {
#     'encoder' : load_model_path_base+'_encoder',
#     'decoder' : load_model_path_base+'_attention_decoder'
# }


N_emgsensor = 4
N_fingerAngle = 14
hidden_size = 256
time_length = 39


## load old attention model
#encoder1 = EncoderRNN(N_emgsensor, hidden_size).to(device)
#attn_decoder1 = AttnDecoderRNN(hidden_size, N_fingerAngle, dropout_p=0.1,time_length=time_length,).to(device)

#encoder1.load_state_dict(torch.load(load_model_path_list['encoder']))
#attn_decoder1.load_state_dict(torch.load(load_model_path_list['decoder']))


#encoder1.eval()
#attn_decoder1.eval()


## load test result
eval_decoder_attention_19 = np.load(load_model_path_base_19+'/eval_decoder_attention.npy')
test_decoder_attention_19 = np.load(load_model_path_base_19+'/test_decoder_attention.npy')
eval_decoder_attention_29 = np.load(load_model_path_base_29+'/eval_decoder_attention.npy')
test_decoder_attention_29 = np.load(load_model_path_base_29+'/test_decoder_attention.npy')
eval_decoder_attention_39 = np.load(load_model_path_base_39+'/eval_decoder_attention.npy')
test_decoder_attention_39 = np.load(load_model_path_base_39+'/test_decoder_attention.npy')
eval_decoder_attention_49 = np.load(load_model_path_base_49+'/eval_decoder_attention.npy')
test_decoder_attention_49 = np.load(load_model_path_base_49+'/test_decoder_attention.npy')
eval_decoder_attention_59 = np.load(load_model_path_base_59+'/eval_decoder_attention.npy')
test_decoder_attention_59 = np.load(load_model_path_base_59+'/test_decoder_attention.npy')
eval_decoder_attention_69 = np.load(load_model_path_base_69+'/eval_decoder_attention.npy')
test_decoder_attention_69 = np.load(load_model_path_base_69+'/test_decoder_attention.npy')
eval_decoder_attention_79 = np.load(load_model_path_base_79+'/eval_decoder_attention.npy')
test_decoder_attention_79 = np.load(load_model_path_base_79+'/test_decoder_attention.npy')
eval_decoder_attention_89 = np.load(load_model_path_base_89+'/eval_decoder_attention.npy')
test_decoder_attention_89 = np.load(load_model_path_base_89+'/test_decoder_attention.npy')

## load test result for c

eval_decoder_attention_new2  = np.load('/home/hyuninlee/PycharmProjects/xcorps/seq2seq_attentionmodel/result/Attention_new/'
                                  'lre_0.0004_lrd_0.0005_bs_1024_tfr_0.5_tl_1__05-07-2021_00-19-26/best_eval_attention_scores.npy')


#test_pred_data_attention = np.load(load_model_path_base+'/test_pred_target.npy')
#test_pred_data_noattention = np.load(load_path_noattentionmodel)
#test_pred_data_simpleNN = np.load(load_path_simpleNN)

##load train,eval,test data
input_data, output_data,input_data_eval,output_data_eval = dataprepare(data_path,doesEval=True)
test_input_data, test_output_data, _, _ = dataprepare(test_path, test=True)

# test_output_data convert 0 <-> 1
output_data = 1-output_data
output_data_eval = 1-output_data_eval
test_output_data = 1 - test_output_data

def showAttentionALL2(att1,str1,att2,str2,att3,str3,att4,str4,att5,str5):
    fig, axes = plt.subplots(nrows=5, ncols=1,gridspec_kw={'height_ratios': [1, 1,1,1,1]})
    att = [att1[:,:200],att2[:,:200],att3[:,:200],att4[:,:200],att5[:,:200]]
    str = [str1,str2,str3,str4,str5]
    i=0
    for ax in axes.flat:
        #ax.set_axis_off()
        im = ax.matshow(att[i], cmap='bone')
        ax.title.set_text('time length = ' + str[i])
        i+=1

    # fig.subplots_adjust(bottom=0.1, top=0.9, left=0.1, right=0.8,
    #                     wspace=0.02, hspace=0.02)

    # add an axes, lower left corner in [0.83, 0.1] measured in figure coordinate with axes width 0.02 and height 0.8

    cb_ax = fig.add_axes([0.83, 0.1, 0.02, 0.8])
    cbar = fig.colorbar(im,cax = cb_ax)

    #set the colorbar ticks and tick labels
    #cbar.set_ticks(np.arange(0, 1.1, 0.5))
    #cbar.set_ticklabels(['low', 'medium', 'high'])

    plt.show()


def showAttentionALL(att1,str1,att2,str2,att3,str3,att4,str4,att5,str5):

    att1 = att1[:,:200]
    att2 = att2[:, :200]
    att3 = att3[:, :200]
    att4 = att4[:, :200]
    att5 = att5[:,:200]
    # Set up figure with colorbar
    fig = plt.figure()
    fig.suptitle('attention map')
    ax1 = fig.add_subplot(511)
    ax1.title.set_text('time length = '+ str1)
    ax2 = fig.add_subplot(512)
    ax2.title.set_text('time length = '+ str2)
    ax3 = fig.add_subplot(513)
    ax3.title.set_text('time length = '+ str3)
    ax4 = fig.add_subplot(514)
    ax4.title.set_text('time length = '+ str4)
    ax5 = fig.add_subplot(515)
    ax5.title.set_text('time length = '+ str5)


    fig.colorbar(ax1matshow(att1, cmap='bone'))
    cax2 = ax1.matshow(att2, cmap='bone')
    fig.colorbar(cax2)
    cax3 = ax1.matshow(att3, cmap='bone')
    fig.colorbar(cax3)
    cax4 = ax1.matshow(att4, cmap='bone')
    fig.colorbar(cax4)
    cax5 = ax1.matshow(att5, cmap='bone')
    fig.colorbar(cax5)

    EMGSENSOR = ['A', 'B', 'C', 'D', 'E']


    # Set up axes
    #ax.set_xticklabels([''] + EMGSENSOR, rotation=90)
    #ax.set_yticklabels([''] + output_words)

    # Show label at every tick
    #ax.xaxis.set_major_locator(ticker.MultipleLocator(1))
    #ax.yaxis.set_major_locator(ticker.MultipleLocator(1))

    plt.show()


def showAttention(emgvalue, attentions,str_timelength,title):

    emgvalue = emgvalue[:,:200]
    attentions = attentions[:,:200]
    maxtime = np.argmax(attentions, axis=0)
    # Set up figure with colorbar
    assert emgvalue.shape[0] == 4
    fig = plt.figure()
    fig.suptitle(title+' : attention map(time length = '+str_timelength+')')
    ax = fig.add_subplot(411)
    cax = ax.matshow(attentions, cmap='bone')
    fig.colorbar(cax)

    EMGSENSOR = ['A', 'B', 'C', 'D', 'E']


    # Set up axes
    #ax.set_xticklabels([''] + EMGSENSOR, rotation=90)
    #ax.set_yticklabels([''] + output_words)

    # Show label at every tick
    #ax.xaxis.set_major_locator(ticker.MultipleLocator(1))
    #ax.yaxis.set_major_locator(ticker.MultipleLocator(1))

    plt.show()
    return maxtime

def showAttention_fornew(emgvalue, attentions,title):

    # attentions_forthumb = attentions[0:4,:]
    # attentions_forindex = attentions[4:8, :]
    # attentions_formiddle = attentions[8:12, :]
    # attentions_forring = attentions[12:16, :]
    # attentions_forpinky = attentions[16:20, :]

    attentions[[4 * 1 + 0, 4 * 1 + 1]] = attentions[[4 * 1 + 1, 4 * 1 + 0]]
    attentions[[4 * 10 + 1, 4 * 10 + 2]] = attentions[[4 * 10 + 2, 4 * 10 + 1]]


    maxtime = np.argmax(attentions, axis=0)
    # Set up figure with colorbar
    assert emgvalue.shape[0] == 4
    fig = plt.figure(figsize=(16.0, 10.0))
    fig.suptitle(title)
    fig.tight_layout()

    ax1 = plt.subplot(14,1,1)
    ax2 = plt.subplot(14,1,2)
    ax3 = plt.subplot(14,1,3)
    ax4 = plt.subplot(14,1,4)
    ax5 = plt.subplot(14,1,5)
    ax6 = plt.subplot(14,1,6)
    ax7 = plt.subplot(14,1,7)
    ax8 = plt.subplot(14,1,8)
    ax9 = plt.subplot(14,1,9)
    ax10 = plt.subplot(14,1,10)
    ax11 = plt.subplot(14,1,11)
    ax12 = plt.subplot(14,1,12)
    ax13 = plt.subplot(14,1,13)
    ax14 = plt.subplot(14,1,14)

    axes = [ax1,ax2,ax3,ax4,ax5,ax6,ax7,ax8,ax9,ax10,ax11,ax12,ax13,ax14]



    cax1 = ax1.matshow(attentions[4*0:4*1,:], cmap='bone')
    cax2 = ax2.matshow(attentions[4*1:4*2,:], cmap='bone')
    cax3 = ax3.matshow(attentions[4*2:4*3,:], cmap='bone')
    cax4 = ax4.matshow(attentions[4*3:4*4,:], cmap='bone')
    cax5 = ax5.matshow(attentions[4*4:4*5,:], cmap='bone')
    cax6 = ax6.matshow(attentions[4*5:4*6,:], cmap='bone')
    cax7 = ax7.matshow(attentions[4*6:4*7,:], cmap='bone')
    cax8 = ax8.matshow(attentions[4*7:4*8,:], cmap='bone')
    cax9 = ax9.matshow(attentions[4*8:4*9,:], cmap='bone')
    cax10 = ax10.matshow(attentions[4*9:4*10,:], cmap='bone')
    cax11 = ax11.matshow(attentions[4*10:4*11,:], cmap='bone')
    cax12 = ax12.matshow(attentions[4*11:4*12,:], cmap='bone')
    cax13 = ax13.matshow(attentions[4*12:4*13,:], cmap='bone')
    cax14 = ax14.matshow(attentions[4*13:4*14,:], cmap='bone')

    for ax_ in axes :
        ax_.set_xticks([])
        ax_.set_yticks([])


    #fig.colorbar(cax5)


    EMGSENSOR = ['A', 'B', 'C', 'D', 'E']


    # Set up axes
    #ax.set_xticklabels([''] + EMGSENSOR, rotation=90)
    #ax.set_yticklabels([''] + output_words)

    # Show label at every tick
    #ax.xaxis.set_major_locator(ticker.MultipleLocator(1))
    #ax.yaxis.set_major_locator(ticker.MultipleLocator(1))

    #plt.show()

    return fig



def kl_divergence(p, q):
    p = p + 1e-6
    q = q + 1e-6
    a = p*np.log(p/q)
    return np.sum(p * np.log(p/q), 0)

def mse(y,t) :
    return np.sqrt((1/2)*np.mean((y-t)**2))



def gettestACC(y,t) :
    sum = 0
    for idx in range(14):
        sum += mse(y[idx, :], t[idx, :])
    return sum/14


if __name__ == "__main__" :

    #smooth_test_pred_data_attention = np.zeros_like(output_data_eval)
    #ComparePlotAngles(test_output_data, test_pred_data_attention)

    ## 1. ICA result of train data( individual data)
    #ica = FastICA(n_components=N_emgsensor)
    #signal_reconstruct_ica = ica.fit_transform(test_output_data)
    #weight_ica = ica.mixing_

    ## 2. attention result of new model

    # for idx in range(14) :
    #     print("=====================")
    #     print("attention model acc : %.3f " %(mse(test_pred_data_attention[idx,:],test_output_data[idx,:])))
    #     print("noattention model acc : %.3f " % (mse(test_pred_data_noattention[idx, :], test_output_data[idx, :])))
    #     print("simpleNN model : %.3f " % (mse(test_pred_data_simpleNN[idx, :], test_output_data[idx, :])))
    #     print("=====================")
    # smooth_test_pred_data_attention = np.zeros_like(test_pred_data_attention)
    # for i in range(14) :
    #     smooth_test_pred_data_attention[i, :] = savgol_filter(test_pred_data_attention[i, :], 59, 5)

    # maxtime19 = showAttention(input_data_eval, eval_decoder_attention_19, '19','eval')
    # maxtime19 = showAttention(input_data_eval, test_decoder_attention_19, '19','test')
    # maxtime29 = showAttention(input_data_eval,eval_decoder_attention_29,'29','eval')
    # maxtime39 = showAttention(input_data_eval, eval_decoder_attention_39,'39','eval')
    # maxtime49 = showAttention(input_data_eval, eval_decoder_attention_49,'49','eval')
    # maxtime59 = showAttention(input_data_eval, eval_decoder_attention_59,'59','eval')
    # maxtime69 = showAttention(input_data_eval,eval_decoder_attention_69,'69','eval')
    # maxtime79 = showAttention(input_data_eval, eval_decoder_attention_79, '79','eval')
    # maxtime89 = showAttention(input_data_eval, eval_decoder_attention_89, '89','eval')

    # attentions_movethumb = eval_decoder_attention_new2[:, 199*0:199*1]
    # attentions_moveindex = eval_decoder_attention_new2[:, 199*1:199*2]
    # attentions_movemiddle = eval_decoder_attention_new2[:, 199*2:199*3]
    # attentions_movering = eval_decoder_attention_new2[:, 199*3:199*4]
    # attentions_movepinky = eval_decoder_attention_new2[:, 199*4:199*5]
    #
    # showAttention_fornew(input_data_eval,attentions_movethumb,'eval : move thumb')
    # showAttention_fornew(input_data_eval, attentions_moveindex, 'eval : move middle')
    # showAttention_fornew(input_data_eval, attentions_movemiddle, 'eval : move pinky')
    # showAttention_fornew(input_data_eval, attentions_movering, 'eval : move ring')
    # showAttention_fornew(input_data_eval, attentions_movepinky, 'eval : move index')
    #
    # plt.show()

    for i in range(1,101) :
        print(i)
        filename = 'epoch_'+str(i)+'_eval_attention_scores'

        eval_attention_score_astimegoesby = np.load(
            '/home/hyuninlee/PycharmProjects/xcorps/seq2seq_attentionmodel/result/Attention_new/'
            'lre_0.0004_lrd_0.0005_bs_1024_tfr_0.5_tl_1__05-07-2021_00-19-26/'+filename+'.npy')

        attentions_movethumb = eval_attention_score_astimegoesby[:, 199 * 0:199 * 1]
        attentions_moveindex = eval_attention_score_astimegoesby[:, 199 * 1:199 * 2]
        attentions_movemiddle = eval_attention_score_astimegoesby[:, 199 * 2:199 * 3]
        attentions_movering = eval_attention_score_astimegoesby[:, 199 * 3:199 * 4]
        attentions_movepinky = eval_attention_score_astimegoesby[:, 199 * 4:199 * 5]
        ## problem arise because the result matrix is so much long,, so what I have to do is to downsample the matrix and redefine
        attention_length = attentions_movethumb.shape[1]
        ########################################3
        whichOneYouUse = attentions_movepinky
        whichfinger = 'pinky'
        ####################################
        activateonce = True
        for j in range(attention_length) :
            assert attention_length == whichOneYouUse.shape[1]
            if j % 10 == 0  :
                if activateonce == True :
                    whichOneYouUse_new = whichOneYouUse[:,j:j+1]
                    activateonce = False
                else :
                    whichOneYouUse_new = np.concatenate((whichOneYouUse_new,whichOneYouUse[:,j:j+1]), axis=1)



        #fig1 = showAttention_fornew(input_data_eval, attentions_movethumb, 'eval : move thumb / epoch : '+str(i))
        fig1 = showAttention_fornew(input_data_eval, whichOneYouUse_new, 'eval : move '+whichfinger+' / epoch : ' + str(i))
        fig1.savefig('/home/hyuninlee/PycharmProjects/xcorps/seq2seq_attentionmodel/result/Attention_new/'
            'lre_0.0004_lrd_0.0005_bs_1024_tfr_0.5_tl_1__05-07-2021_00-19-26/'+whichfinger+'_'+filename+'.png')
        plt.close()
        #showAttention_fornew(input_data_eval, attentions_moveindex, 'eval : move middle')
        #showAttention_fornew(input_data_eval, attentions_movemiddle, 'eval : move pinky')
        #showAttention_fornew(input_data_eval, attentions_movering, 'eval : move ring')
        #showAttention_fornew(input_data_eval, attentions_movepinky, 'eval : move index')





    # showAttentionALL2(eval_decoder_attention_29,'29', eval_decoder_attention_39,'39',
    #                  eval_decoder_attention_49,'49',eval_decoder_attention_59,'59',
    #                  eval_decoder_attention_59,'69',)

    #ComparePlot3Angles(test_output_data,test_pred_data_simpleNN,'Neural Net',smooth_test_pred_data_attention,'attention' ,)
    # ComparePlotAngles(test_output_data,test_pred_data_attention)
    #ComparePlotAngles(test_output_data,smooth_test_pred_data_attention)