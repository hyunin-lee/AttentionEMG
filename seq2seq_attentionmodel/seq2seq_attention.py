# -*- coding: utf-8 -*-
"""
NLP From Scratch: Translation with a Sequence to Sequence Network and Attention
*******************************************************************************
**Author**: `Sean Robertson <https://github.com/spro/practical-pytorch>`_

This is the third and final tutorial on doing "NLP From Scratch", where we
write our own classes and functions to preprocess the data to do our NLP
modeling tasks. We hope after you complete this tutorial that you'll proceed to
learn how `torchtext` can handle much of this preprocessing for you in the
three tutorials immediately following this one.

In this project we will be teaching a neural network to translate from
French to English.

::

    [KEY: > input, = target, < output]

    > il est en train de peindre un tableau .
    = he is painting a picture .
    < he is painting a picture .

    > pourquoi ne pas essayer ce vin delicieux ?
    = why not try that delicious wine ?
    < why not try that delicious wine ?

    > elle n est pas poete mais romanciere .
    = she is not a poet but a novelist .
    < she not not a poet but a novelist .

    > vous etes trop maigre .
    = you re too skinny .
    < you re all alone .

... to varying degrees of success.

This is made possible by the simple but powerful idea of the `sequence
to sequence network <https://arxiv.org/abs/1409.3215>`__, in which two
recurrent neural networks work together to transform one sequence to
another. An encoder network condenses an input sequence into a vector,
and a decoder network unfolds that vector into a new sequence.

.. figure:: /_static/img/seq-seq-images/seq2seq.png
   :alt:

To improve upon this model we'll use an `attention
mechanism <https://arxiv.org/abs/1409.0473>`__, which lets the decoder
learn to focus over a specific range of the input sequence.

**Recommended Reading:**

I assume you have at least installed PyTorch, know Python, and
understand Tensors:

-  https://pytorch.org/ For installation instructions
-  :doc:`/beginner/deep_learning_60min_blitz` to get started with PyTorch in general
-  :doc:`/beginner/pytorch_with_examples` for a wide and deep overview
-  :doc:`/beginner/former_torchies_tutorial` if you are former Lua Torch user


It would also be useful to know about Sequence to Sequence networks and
how they work:

-  `Learning Phrase Representations using RNN Encoder-Decoder for
   Statistical Machine Translation <https://arxiv.org/abs/1406.1078>`__
-  `Sequence to Sequence Learning with Neural
   Networks <https://arxiv.org/abs/1409.3215>`__
-  `Neural Machine Translation by Jointly Learning to Align and
   Translate <https://arxiv.org/abs/1409.0473>`__
-  `A Neural Conversational Model <https://arxiv.org/abs/1506.05869>`__

You will also find the previous tutorials on
:doc:`/intermediate/char_rnn_classification_tutorial`
and :doc:`/intermediate/char_rnn_generation_tutorial`
helpful as those concepts are very similar to the Encoder and Decoder
models, respectively.

And for more, read the papers that introduced these topics:

-  `Learning Phrase Representations using RNN Encoder-Decoder for
   Statistical Machine Translation <https://arxiv.org/abs/1406.1078>`__
-  `Sequence to Sequence Learning with Neural
   Networks <https://arxiv.org/abs/1409.3215>`__
-  `Neural Machine Translation by Jointly Learning to Align and
   Translate <https://arxiv.org/abs/1409.0473>`__
-  `A Neural Conversational Model <https://arxiv.org/abs/1506.05869>`__


**Requirements**
"""
from __future__ import unicode_literals, print_function, division
from io import open
import unicodedata
import string
import re
import random
import numpy as np

import torch
import torch.nn as nn
from torch import optim
import torch.nn.functional as F

from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
plt.switch_backend('agg')
import matplotlib.ticker as ticker
import numpy as np
from tqdm import tqdm

from torch.utils.tensorboard import SummaryWriter


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

######################################################################
# Loading data files
# ==================
#
# The data for this project is a set of many thousands of English to
# French translation pairs.
#
# `This question on Open Data Stack
# Exchange <https://opendata.stackexchange.com/questions/3888/dataset-of-sentences-translated-into-many-languages>`__
# pointed me to the open translation site https://tatoeba.org/ which has
# downloads available at https://tatoeba.org/eng/downloads - and better
# yet, someone did the extra work of splitting language pairs into
# individual text files here: https://www.manythings.org/anki/
#
# The English to French pairs are too big to include in the repo, so
# download to ``data/eng-fra.txt`` before continuing. The file is a tab
# separated list of translation pairs:
#
# ::
#
#     I am cold.    J'ai froid.
#
# .. Note::
#    Download the data from
#    `here <https://download.pytorch.org/tutorial/data.zip>`_
#    and extract it to the current directory.

######################################################################
# Similar to the character encoding used in the character-level RNN
# tutorials, we will be representing each word in a language as a one-hot
# vector, or giant vector of zeros except for a single one (at the index
# of the word). Compared to the dozens of characters that might exist in a
# language, there are many many more words, so the encoding vector is much
# larger. We will however cheat a bit and trim the data to only use a few
# thousand words per language.
#
# .. figure:: /_static/img/seq-seq-images/word-encoding.png
#    :alt:
#
#


######################################################################
# We'll need a unique index per word to use as the inputs and targets of
# the networks later. To keep track of all this we will use a helper class
# called ``Lang`` which has word → index (``word2index``) and index → word
# (``index2word``) dictionaries, as well as a count of each word
# ``word2count`` to use to later replace rare words.
#

SOS_token = 0
EOS_token = 1


class Lang:
    def __init__(self, name):
        self.name = name
        self.word2index = {}
        self.word2count = {}
        self.index2word = {0: "SOS", 1: "EOS"}
        self.n_words = 2  # Count SOS and EOS

    def addSentence(self, sentence):
        for word in sentence.split(' '):
            self.addWord(word)

    def addWord(self, word):
        if word not in self.word2index:
            self.word2index[word] = self.n_words
            self.word2count[word] = 1
            self.index2word[self.n_words] = word
            self.n_words += 1
        else:
            self.word2count[word] += 1


######################################################################
# The files are all in Unicode, to simplify we will turn Unicode
# characters to ASCII, make everything lowercase, and trim most
# punctuation.
#

# Turn a Unicode string to plain ASCII, thanks to
# https://stackoverflow.com/a/518232/2809427
def unicodeToAscii(s):
    return ''.join(
        c for c in unicodedata.normalize('NFD', s)
        if unicodedata.category(c) != 'Mn'
    )

# Lowercase, trim, and remove non-letter characters


def normalizeString(s):
    s = unicodeToAscii(s.lower().strip())
    s = re.sub(r"([.!?])", r" \1", s)
    s = re.sub(r"[^a-zA-Z.!?]+", r" ", s)
    return s


######################################################################
# To read the data file we will split the file into lines, and then split
# lines into pairs. The files are all English → Other Language, so if we
# want to translate from Other Language → English I added the ``reverse``
# flag to reverse the pairs.
#

def readLangs(lang1, lang2, reverse=False):
    print("Reading lines...")

    # Read the file and split into lines
    lines = open('data/%s-%s.txt' % (lang1, lang2), encoding='utf-8').\
        read().strip().split('\n')
    # lines : {list : 135842} lines =[ 'Go.\tVa!', 'Run!\tCours\u202f!",.... ]
    # Split every line into pairs and normalize
    pairs = [[normalizeString(s) for s in l.split('\t')] for l in lines]
    # pairs = [['go.','va!'],['run!,'cours!'],...]
    # Reverse pairs, make Lang instances
    if reverse:
        pairs = [list(reversed(p)) for p in pairs]
        # pairs = [['va!', go.'],['cours!','run!],...]
        input_lang = Lang(lang2)
        output_lang = Lang(lang1)
    else:
        input_lang = Lang(lang1)
        output_lang = Lang(lang2)

    return input_lang, output_lang, pairs


######################################################################
# Since there are a *lot* of example sentences and we want to train
# something quickly, we'll trim the data set to only relatively short and
# simple sentences. Here the maximum length is 10 words (that includes
# ending punctuation) and we're filtering to sentences that translate to
# the form "I am" or "He is" etc. (accounting for apostrophes replaced
# earlier).
#

MAX_LENGTH = 30

# eng_prefixes = (
#     "i am ", "i m ",
#     "he is", "he s ",
#     "she is", "she s ",
#     "you are", "you re ",
#     "we are", "we re ",
#     "they are", "they re "
# )


def filterPair(p):
    return len(p[0].split(' ')) < MAX_LENGTH and \
        len(p[1].split(' ')) < MAX_LENGTH and \
        p[1].startswith(eng_prefixes)


def filterPairs(pairs):
    return [pair for pair in pairs if filterPair(pair)]


######################################################################
# The full process for preparing the data is:
#
# -  Read text file and split into lines, split lines into pairs
# -  Normalize text, filter by length and content
# -  Make word lists from sentences in pairs
#

def prepareData(lang1, lang2, reverse=False):
    input_lang, output_lang, pairs = readLangs(lang1, lang2, reverse)
    print("Read %s sentence pairs" % len(pairs))
    pairs = filterPairs(pairs)
    print("Trimmed to %s sentence pairs" % len(pairs))
    print("Counting words...")
    for pair in pairs:
        input_lang.addSentence(pair[0])
        output_lang.addSentence(pair[1])
    print("Counted words:")
    print(input_lang.name, input_lang.n_words)
    print(output_lang.name, output_lang.n_words)
    return input_lang, output_lang, pairs


# input_lang, output_lang, pairs = prepareData('eng', 'fra', True)
# print(random.choice(pairs))
"""
hyunin comments : how does input_lang, output_lang looks like? 
input_lang = class
    index2word = {dict : 4345} {0: 'SOS', 1: 'EOS', 2: 'j', 3: 'ai', 4: 'ans', 5: '.', 6: 'je'
    n_words = 4345
    name = 'fra'
    word2count = {dict:4345} {'j': 414, 'ai': 340, 'ans': 55, '.': 10262, 'je': 3654
    word2index = {dict:4345} {'j': 2, 'ai': 3, 'ans': 4, '.': 5, 'je': 6, 'vais': 7, 'bien': 8

output_lang = class
    index2word = {dict : 2803} {0: 'SOS', 1: 'EOS', 2: 'i', 3: 'm', 4: '.', 5: 'ok',
    n_words = 2803
    name = 'eng'
    word2count = {dict:2801} 
    word2index = {dict:2801} 

"""




######################################################################
# The Seq2Seq Model
# =================
#
# A Recurrent Neural Network, or RNN, is a network that operates on a
# sequence and uses its own output as input for subsequent steps.
#
# A `Sequence to Sequence network <https://arxiv.org/abs/1409.3215>`__, or
# seq2seq network, or `Encoder Decoder
# network <https://arxiv.org/pdf/1406.1078v3.pdf>`__, is a model
# consisting of two RNNs called the encoder and decoder. The encoder reads
# an input sequence and outputs a single vector, and the decoder reads
# that vector to produce an output sequence.
#
# .. figure:: /_static/img/seq-seq-images/seq2seq.png
#    :alt:
#
# Unlike sequence prediction with a single RNN, where every input
# corresponds to an output, the seq2seq model frees us from sequence
# length and order, which makes it ideal for translation between two
# languages.
#
# Consider the sentence "Je ne suis pas le chat noir" → "I am not the
# black cat". Most of the words in the input sentence have a direct
# translation in the output sentence, but are in slightly different
# orders, e.g. "chat noir" and "black cat". Because of the "ne/pas"
# construction there is also one more word in the input sentence. It would
# be difficult to produce a correct translation directly from the sequence
# of input words.
#
# With a seq2seq model the encoder creates a single vector which, in the
# ideal case, encodes the "meaning" of the input sequence into a single
# vector — a single point in some N dimensional space of sentences.
#


######################################################################
# The Encoder
# -----------
#
# The encoder of a seq2seq network is a RNN that outputs some value for
# every word from the input sentence. For every input word the encoder
# outputs a vector and a hidden state, and uses the hidden state for the
# next input word.
#
# .. figure:: /_static/img/seq-seq-images/encoder-network.png
#    :alt:
#
#

class EncoderRNN(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(EncoderRNN, self).__init__()
        self.hidden_size = hidden_size

        self.embedding = nn.Embedding(input_size, hidden_size)
        self.gru = nn.GRU(hidden_size, hidden_size)

    def forward(self, input, hidden):
        embedded = self.embedding(input).view(1, 1, -1)
        output = embedded
        output, hidden = self.gru(output, hidden)
        return output, hidden

    def initHidden(self):
        return torch.zeros(1, 1, self.hidden_size, device=device)

######################################################################
# The Decoder
# -----------
#
# The decoder is another RNN that takes the encoder output vector(s) and
# outputs a sequence of words to create the translation.
#


######################################################################
# Simple Decoder
# ^^^^^^^^^^^^^^
#
# In the simplest seq2seq decoder we use only last output of the encoder.
# This last output is sometimes called the *context vector* as it encodes
# context from the entire sequence. This context vector is used as the
# initial hidden state of the decoder.
#
# At every step of decoding, the decoder is given an input token and
# hidden state. The initial input token is the start-of-string ``<SOS>``
# token, and the first hidden state is the context vector (the encoder's
# last hidden state).
#
# .. figure:: /_static/img/seq-seq-images/decoder-network.png
#    :alt:
#
#

class DecoderRNN(nn.Module):
    def __init__(self, hidden_size, output_size):
        super(DecoderRNN, self).__init__()
        self.hidden_size = hidden_size

        self.embedding = nn.Embedding(output_size, hidden_size)
        self.gru = nn.GRU(hidden_size, hidden_size)
        self.out = nn.Linear(hidden_size, output_size)
        self.softmax = nn.LogSoftmax(dim=1)

    def forward(self, input, hidden):
        output = self.embedding(input).view(1, 1, -1)
        output = F.relu(output)
        output, hidden = self.gru(output, hidden)
        output = self.sigmoid(self.out(output[0]))
        return output, hidden

    def initHidden(self):
        return torch.zeros(1, 1, self.hidden_size, device=device)

######################################################################
# I encourage you to train and observe the results of this model, but to
# save space we'll be going straight for the gold and introducing the
# Attention Mechanism.
#


######################################################################
# Attention Decoder
# ^^^^^^^^^^^^^^^^^
#
# If only the context vector is passed betweeen the encoder and decoder,
# that single vector carries the burden of encoding the entire sentence.
#
# Attention allows the decoder network to "focus" on a different part of
# the encoder's outputs for every step of the decoder's own outputs. First
# we calculate a set of *attention weights*. These will be multiplied by
# the encoder output vectors to create a weighted combination. The result
# (called ``attn_applied`` in the code) should contain information about
# that specific part of the input sequence, and thus help the decoder
# choose the right output words.
#
# .. figure:: https://i.imgur.com/1152PYf.png
#    :alt:
#
# Calculating the attention weights is done with another feed-forward
# layer ``attn``, using the decoder's input and hidden state as inputs.
# Because there are sentences of all sizes in the training data, to
# actually create and train this layer we have to choose a maximum
# sentence length (input length, for encoder outputs) that it can apply
# to. Sentences of the maximum length will use all the attention weights,
# while shorter sentences will only use the first few.
#
# .. figure:: /_static/img/seq-seq-images/attention-decoder-network.png
#    :alt:
#
#

class AttnDecoderRNN(nn.Module):
    def __init__(self, hidden_size, output_size, dropout_p=0.1, max_length=MAX_LENGTH):
        super(AttnDecoderRNN, self).__init__()
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.dropout_p = dropout_p
        self.max_length = max_length

        self.embedding = nn.Embedding(self.output_size, self.hidden_size)
        self.attn = nn.Linear(self.hidden_size * 2, self.max_length)
        self.attn_combine = nn.Linear(self.hidden_size * 2, self.hidden_size)
        self.dropout = nn.Dropout(self.dropout_p)
        self.gru = nn.GRU(self.hidden_size, self.hidden_size)
        self.out = nn.Linear(self.hidden_size, int(self.output_size *0.5))

    def forward(self, input, hidden, encoder_outputs):
        embedded = self.embedding(input).view(1, 1, -1) #input: torch.size([1,1]) embedded : (1,1,128)
        embedded = self.dropout(embedded)

        attn_weights = F.softmax(
            self.attn(torch.cat((embedded[0], hidden[0]), 1)), dim=1)
        #embedded[0] : (1,128) #hidden[0] : (1,128) #torch.cat() : (1,256)
        #softmax's dim =1 means applying soft max over "256"
        #attn_weights : (1,10=max_length)
        attn_applied = torch.bmm(attn_weights.unsqueeze(0),
                                 encoder_outputs.unsqueeze(0))
        #attn_weights.unsqueeze(0) : (1,10) -> (1,1,10)
        #encoder_outputs.unsqueeze(0) : (10,128) -> (1,10,128)
        #attn_applied : (1,1,128)
        output = torch.cat((embedded[0], attn_applied[0]), 1)
        #output : (1,256=2*hs)
        output = self.attn_combine(output).unsqueeze(0)
        # output : (1,1,hs)

        output = F.relu(output)  # output : (1,1,hs)
        output, hidden = self.gru(output, hidden)
        # output : (1,1,hs) , hidden: (1,1,hs)
        output = F.sigmoid(self.out(output[0]))
        # ouput : (1,2803)
        return output, hidden, attn_weights

    def initHidden(self):
        return torch.zeros(1, 1, self.hidden_size, device=device)


######################################################################
# .. note:: There are other forms of attention that work around the length
#   limitation by using a relative position approach. Read about "local
#   attention" in `Effective Approaches to Attention-based Neural Machine
#   Translation <https://arxiv.org/abs/1508.04025>`__.
#
# Training
# ========
#
# Preparing Training Data
# -----------------------
#
# To train, for each pair we will need an input tensor (indexes of the
# words in the input sentence) and target tensor (indexes of the words in
# the target sentence). While creating these vectors we will append the
# EOS token to both sequences.
#

def indexesFromSentence(lang, sentence):
    return [lang.word2index[word] for word in sentence.split(' ')]


def tensorFromSentence(lang, sentence):
    indexes = indexesFromSentence(lang, sentence) #indexes [118,214,188,2030,5]
    indexes.append(EOS_token) #indexes [118,214,188,2030,5,1]
    return torch.tensor(indexes, dtype=torch.long, device=device).view(-1, 1) #tensor([[ 118],[ 214],[ 188],[2030],[   5],[   1]], device='cuda:0')
    #retunr all the other onseuqencse

def tensorsFromPair(pair):
    input_tensor = tensorFromSentence(input_lang, pair[0])
    target_tensor = tensorFromSentence(output_lang, pair[1])
    # pair :['je souffre d un mauvais rhume .', 'i am suffering from a bad cold .']
    # tensor([[   6],
    #         [ 414],
    #         [ 233],
    #         [  66],
    #         [ 350],
    #         [3416],
    #         [   5],
    #         [   1]], device='cuda:0')

    # tensor([[   2],
    #         [  16],
    #         [1953],
    #         [ 499],
    #         [  42],
    #         [ 130],
    #         [  21],
    #         [   4],
    #         [   1]], device='cuda:0')

    return (input_tensor, target_tensor)


######################################################################
# Training the Model
# ------------------
#
# To train we run the input sentence through the encoder, and keep track
# of every output and the latest hidden state. Then the decoder is given
# the ``<SOS>`` token as its first input, and the last hidden state of the
# encoder as its first hidden state.
#
# "Teacher forcing" is the concept of using the real target outputs as
# each next input, instead of using the decoder's guess as the next input.
# Using teacher forcing causes it to converge faster but `when the trained
# network is exploited, it may exhibit
# instability <http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.378.4095&rep=rep1&type=pdf>`__.
#
# You can observe outputs of teacher-forced networks that read with
# coherent grammar but wander far from the correct translation -
# intuitively it has learned to represent the output grammar and can "pick
# up" the meaning once the teacher tells it the first few words, but it
# has not properly learned how to create the sentence from the translation
# in the first place.
#
# Because of the freedom PyTorch's autograd gives us, we can randomly
# choose to use teacher forcing or not with a simple if statement. Turn
# ``teacher_forcing_ratio`` up to use more of it.
#

teacher_forcing_ratio = 1.0


def train(input_tensor, target_tensor, encoder, decoder, encoder_optimizer, decoder_optimizer, criterion, max_length=MAX_LENGTH):

    encoder_hidden = encoder.initHidden()

    encoder_optimizer.zero_grad()
    decoder_optimizer.zero_grad()

    input_length = input_tensor.size(0)
    target_length = target_tensor.size(0)

    decoder_save = torch.zeros((target_length, target_length))

    assert max_length == input_length and max_length == target_length
    encoder_outputs = torch.zeros(max_length, encoder.hidden_size, device=device)
    input_tensor = input_tensor.to(device)
    target_tensor = target_tensor.to(device)
    loss = 0

    for ei in range(input_length):
        encoder_output, encoder_hidden = encoder(
            input_tensor[ei], encoder_hidden)
        encoder_outputs[ei] = encoder_output[0, 0]

        # input_tensor = torch.Size([7,1])
        # encoder_hidden = torch.Size([1,1,256])
        # encoder_output = torch.Size([1,1,256])
        # encoder_output[ei] = torch.Size([256])

    decoder_input = input_tensor
    #decoder_input = torch.tensor([[SOS_token]], device=device)
    #decoder_input = tensor([[0]], device='cuda:0')
    decoder_hidden = encoder_hidden
    #decoder_hidden = torch.Size([1, 1, 256])
    # use_teacher_forcing = True if random.random() < teacher_forcing_ratio else False
    # there should be  teacher forcing.
    use_teacher_forcing = True
    if use_teacher_forcing:
        # Teacher forcing: Feed the target as the next input
        for di in range(target_length):
            decoder_output, decoder_hidden, decoder_attention = decoder(
                decoder_input[di], decoder_hidden, encoder_outputs)
            # decoder_output : torch.Size([1,ouput_lang.nwords = 2803])
            # decoder_hidden : torch.Size([1,1,hidden_size = 256])
            # decoder_output : torch.Size([1,max_length=10])
            decoder_save[:,di] = decoder_attention
            loss += criterion(decoder_output[:,di].to(torch.double), target_tensor[di].to(torch.double))
            # decoder_input = target_tensor[di]  # Teacher forcing

    else:
        # Without teacher forcing: use its own predictions as the next input
        for di in range(target_length):
            decoder_output, decoder_hidden, decoder_attention = decoder(decoder_input, decoder_hidden, encoder_outputs)
            #decoder_output : (1,2803)
            #decoder_hidden : (1,1,hs)
            #decoder_attention : (1,maxlength)
            topv, topi = decoder_output.topk(1)
            decoder_input = topi.squeeze().detach()  # detach from history as input

            loss += criterion(decoder_output, target_tensor[di].to(torch.double))
            if decoder_input.item() == EOS_token:
                break

    loss.backward()

    encoder_optimizer.step()
    decoder_optimizer.step()

    return loss.item() / target_length , decoder_save


######################################################################
# This is a helper function to print time elapsed and estimated time
# remaining given the current time and progress %.
#

import time
import math


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

DIC_VEC2TUPLE = {}

def dataprepare2_forKET(_datapath) :

    _data = np.load(_datapath)
    #######################################
    _inputdata = _data
    ######################################

    ############################################
    _outputdata = _data
    ############################################
    _new_inputdata = []
    _new_outputdata = []
    train_pair , eval_pair , total_pair = [], [] , []

    # now convert input to neww index vector :
    for i in range(_data.shape[0]) :
        temp_tensor_input, temp_tensor_output = [] , []
        for idx,cor in enumerate(_inputdata[i,:]) :
            DIC_VEC2TUPLE[int(cor + 2 * idx)] = (idx,"O" if int(cor) == 1 else "X")
            temp_tensor_input.append(int(cor + 2 * idx))
            temp_tensor_output.append(cor)
        _new_inputdata.append(torch.unsqueeze(torch.tensor(temp_tensor_input),dim=-1))
        _new_outputdata.append(torch.unsqueeze(torch.tensor(temp_tensor_output),dim=-1))

    random.shuffle(total_pair)
    for t_i, t_o in zip(_new_inputdata,_new_outputdata) :
        total_pair.append((t_i,t_o))

    #randomly split the data
    train_pair , eval_pair = train_test_split(total_pair, test_size=0.2)

    return train_pair , eval_pair


def trainIters(encoder, decoder, n_iters, print_every=1000, eval_every=1000, learning_rate=0.001 , epoch = 5):
    start = time.time()
    plot_losses = []
    print_loss_total = 0  # Reset every print_every
    plot_loss_total = 0  # Reset every eval_every


    encoder_optimizer = optim.Adam(encoder.parameters(), lr=learning_rate)
    decoder_optimizer = optim.Adam(decoder.parameters(), lr=learning_rate)
    training_pairs, eval_pairs = dataprepare2_forKET("data.npy")
    ### change the number of iteration ###
    # n_iters = len(training_pairs)
    criterion = nn.BCELoss()
    #######################################
    att_list = []
    input_list = []
    for _ in range(epoch) :
        random.shuffle(training_pairs)
        for iter in tqdm(range(1, n_iters + 1)):
            training_pair = training_pairs[iter - 1]
            input_tensor = training_pair[0]
            target_tensor = training_pair[1]

            loss , attention = train(input_tensor, target_tensor, encoder,
                         decoder, encoder_optimizer, decoder_optimizer, criterion)
            print_loss_total += loss
            plot_loss_total += loss

            if iter % print_every == 0:
                print_loss_avg = print_loss_total / print_every
                print_loss_total = 0
                print('%s (%d %d%%) %.4f' % (timeSince(start, iter / n_iters),
                                             iter, iter / n_iters * 100, print_loss_avg))

            if iter % eval_every == 0:
                print("===== Eval ====")
                ACC = []
                Eval_LOSS = []
                low_eval_loss = 10000
                for eval_iter in range(1,len(eval_pairs) + 1):
                    eval_pair = training_pairs[eval_iter - 1]
                    input_tensor_eval = eval_pair[0]
                    target_tensor_eval = eval_pair[1]
                    acc_eval, loss_eval, att_eval = evaluate_KET(encoder, decoder, criterion, input_tensor_eval, target_tensor_eval)
                    Eval_LOSS.append(loss_eval)
                    ACC.append(acc_eval)
                if low_eval_loss > np.mean(Eval_LOSS):
                    print("\nsave model,,,")
                    torch.save(encoder.state_dict(), SAVEPATH+'/encoder')
                    torch.save(decoder.state_dict(), SAVEPATH+'/decoder')
                    low_eval_loss = np.mean(Eval_LOSS)
                print("\neval acc : " , np.mean(ACC))
                print("eval loss : ", np.mean(Eval_LOSS))
                writer.add_scalar('eval acc', np.mean(ACC), _ * n_iters + iter)
                writer.add_scalar('eval loss', np.mean(Eval_LOSS), _ * n_iters + iter)

            writer.add_scalar('training loss', loss, _ * n_iters + iter)



            if _ == epoch -1 and iter > n_iters - 10 :
                att_list.append(attention.cpu().detach().numpy())
                input_list.append(input_tensor.cpu().detach().numpy())

    return att_list , input_list

def CHECKIters(encoder, decoder):

    training_pairs, eval_pairs = dataprepare2_forKET("data.npy")
    total_pairs = training_pairs + eval_pairs
    del training_pairs
    del eval_pairs
    ### change the number of iteration ###
    # n_iters = len(training_pairs)
    criterion = nn.BCELoss()
    #######################################
    att_list = []
    input_list = []
    n_iters = len(total_pairs)
    for iter in tqdm(range(1, n_iters + 1)):
        total_pair = total_pairs[iter - 1]
        input_tensor_eval = total_pair[0]
        target_tensor_eval = total_pair[1]

        _, loss_eval, att_eval = evaluate_KET(encoder, decoder, criterion, input_tensor_eval, target_tensor_eval)

        att_list.append(att_eval.cpu().detach().numpy())
        input_list.append(input_tensor_eval.cpu().detach().numpy())

        # convert att_eval 30 -> 60 size
        # if iter ==200 :
        #     break

    return att_list , input_list

def evaluate_KET(encoder,decoder,criterion,input_tensor,target_tensor) :
    with torch.no_grad() :
        encoder_hidden = encoder.initHidden()
        input_length = input_tensor.size(0)
        target_length = target_tensor.size(0)

        decoder_save = torch.zeros((target_length, target_length))
    
        assert target_length == input_length
        encoder_outputs = torch.zeros(input_length, encoder.hidden_size, device=device)
        input_tensor = input_tensor.to(device)
        target_tensor = target_tensor.to(device)
        loss = 0

        for ei in range(input_length):
            encoder_output, encoder_hidden = encoder(
                input_tensor[ei], encoder_hidden)
            encoder_outputs[ei] = encoder_output[0, 0]

        decoder_input = input_tensor
        decoder_hidden = encoder_hidden

        use_teacher_forcing = True
        Total_count = 0
        if use_teacher_forcing:
            # Teacher forcing: Feed the target as the next input
            for di in range(target_length):
                decoder_output, decoder_hidden, decoder_attention = decoder(
                    decoder_input[di], decoder_hidden, encoder_outputs)
                decoder_save[:, di] = decoder_attention
                loss += criterion(decoder_output[:, di].to(torch.double), target_tensor[di].to(torch.double))
                t = target_tensor[di].to(torch.double)
                o = 0 if decoder_output[:, di].to(torch.double) < 0.5 else 1 
                if t == o :
                    Total_count += 1

        return Total_count / target_length, loss.item()/target_length,  decoder_save

def showAttention_KET(attentions,input, savename):
    # Set up figure with colorbar
    tuple = []
    for i in input :
        tuple.append(DIC_VEC2TUPLE[i.item()])
    fig = plt.figure(figsize = (10,10))
    ax = fig.add_subplot(111)
    cax = ax.matshow(attentions, cmap='bone')
    fig.colorbar(cax)
    ax.set_xticklabels(tuple)
    plt.xticks(rotation=90)
    plt.xticks(fontsize=7)
    plt.yticks(fontsize=7)
    ax.set_yticklabels(tuple)
    # Show label at every tick
    ax.xaxis.set_major_locator(ticker.MultipleLocator(1))
    ax.yaxis.set_major_locator(ticker.MultipleLocator(1))

    plt.savefig(savename)

    #plt.show()

def showAttention(input_sentence, output_words, attentions):
    # Set up figure with colorbar
    fig = plt.figure()
    ax = fig.add_subplot(111)
    cax = ax.matshow(attentions.numpy(), cmap='bone')
    fig.colorbar(cax)

    # Set up axes
    ax.set_xticklabels([''] + input_sentence.split(' ') +
                       ['<EOS>'], rotation=90)
    ax.set_yticklabels([''] + output_words)

    # Show label at every tick
    ax.xaxis.set_major_locator(ticker.MultipleLocator(1))
    ax.yaxis.set_major_locator(ticker.MultipleLocator(1))

    plt.show()


def getmean(list_input) :
    output_array = None
    for i in range(len(list_input)):
        if i == 0 :
            output_array = list_input[i]
        else :
            output_array += list_input[i]

    return output_array/len(list_input)

if __name__ == "__main__" :

    hidden_size = 128 #256
    num_q = 30
    learning_rate = 0.001
    epoch = 1
    save_name = "lr_" + str(learning_rate) + "_hs_" + str(hidden_size)
    loadmodel = True

    SAVEPATH = "./"+save_name
    import os
    if not os.path.exists(SAVEPATH):
        os.makedirs(SAVEPATH)


    encoder1 = EncoderRNN(2*num_q, hidden_size).to(device)
    attn_decoder1 = AttnDecoderRNN(hidden_size, 2*num_q, dropout_p=0.2).to(device)
    if not loadmodel :
        writer = SummaryWriter("./" + save_name)
        att_result_list , input_vector_list = trainIters(encoder1, attn_decoder1, 1400, print_every=100, eval_every=100, learning_rate=learning_rate, epoch =epoch)
        for idx, (att_result , input_vector) in enumerate(zip(att_result_list , input_vector_list)) :
            np.save(SAVEPATH+"/attn_result"+str(idx)+".npy", att_result)
            showAttention_KET(att_result,input_vector,SAVEPATH+"/attn_result"+str(idx)+".png")

    else :
        encoder1.load_state_dict(torch.load(SAVEPATH+"/encoder"))
        attn_decoder1.load_state_dict(torch.load(SAVEPATH + "/decoder"))
        att_eval_list , input_eval_list = CHECKIters(encoder1,attn_decoder1)
        dic = {}
        output_array = np.zeros((2*num_q,2*num_q))
        for i in range(2*num_q) :
            dic[i] = []
        for input_eval, att_eval in zip(input_eval_list,att_eval_list) :
            ## change att_eval 30 -> 60 :
            new_att_eval = np.zeros((2*num_q,2*num_q))
            for idx1,_i1 in enumerate(input_eval) :
                for idx2, _i2 in enumerate(input_eval):
                    new_att_eval[_i1.item(),_i2.item()] = att_eval[idx1,idx2]

            for i_eval in input_eval :
                dic[i_eval.item()].append(new_att_eval[:,i_eval.item()])
        for key,values in dic.items() :
            output_array[:,key] = getmean(values)
        showAttention_KET(output_array,np.arange(start=0, stop=2*num_q-1), SAVEPATH+"/total_att.png")



######################################################################
# For a better viewing experience we will do the extra work of adding axes
# and labels:
#


