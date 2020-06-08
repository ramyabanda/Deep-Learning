#!/usr/bin/env python

##  text_classification_no_gru.py

"""
This script shows how you can use a neural network with feedback for
the classification of a variable-length sequence.  The main idea is to
represent a variable-length input with a fixed-length hidden state 
vector.  

The reason for their being "_no_gru" in the name of the script is that
it does not use any gating mechanisms as a protection against the 
vanishing gradients.
"""

import random
import numpy
import torch
import os, sys


seed = 0           
random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
numpy.random.seed(seed)
torch.backends.cudnn.deterministic=True
torch.backends.cudnn.benchmarks=False
os.environ['PYTHONHASHSEED'] = str(seed)


##  watch -d -n 0.5 nvidia-smi

from DLStudio import *

dls = DLStudio(
#                  dataroot = "/home/kak/TextDatasets/",
                  dataroot = "./data/",
                  path_saved_model = "./saved_model",
                  momentum = 0.9,
                  learning_rate =  0.004,  
                  epochs = 1,
                  batch_size = 1,
                  classes = ('negative','positive'),
                  debug_train = 1,
                  debug_test = 1,
                  use_gpu = True,
              )


text_cl = DLStudio.TextClassification( dl_studio = dls )
dataserver_train = DLStudio.TextClassification.SentimentAnalysisDataset(
                                 train_or_test = 'train',
                                 dl_studio = dls,
#                                dataset_file = "sentiment_dataset_train_3.tar.gz",
                                 dataset_file = "sentiment_dataset_train_200.tar.gz",
#                                dataset_file = "sentiment_dataset_train_40.tar.gz",
                                                                      )
dataserver_test = DLStudio.TextClassification.SentimentAnalysisDataset(
                                 train_or_test = 'test',
                                 dl_studio = dls,
#                                dataset_file = "sentiment_dataset_test_3.tar.gz",
                                 dataset_file = "sentiment_dataset_test_200.tar.gz",
#                                 dataset_file = "sentiment_dataset_test_40.tar.gz",
                                                                  )
text_cl.dataserver_train = dataserver_train
text_cl.dataserver_test = dataserver_test

text_cl.load_SentimentAnalysisDataset(dataserver_train, dataserver_test)

vocab_size = dataserver_train.get_vocab_size()
hidden_size = 512
output_size = 2                            # for positive and negative sentiments

model = text_cl.TEXTnet(vocab_size, hidden_size, output_size)

##  DO NOT UNCOMMENT THE NEXT LINE UNLESS YOU KNOW WHAT YOU ARE DOING.
#model = text_cl.TEXTnetOrder2(vocab_size, hidden_size, output_size, dls)

number_of_learnable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
num_layers = len(list(model.parameters()))

print("\n\nThe number of layers in the model: %d" % num_layers)
print("\nThe number of learnable parameters in the model: %d" % number_of_learnable_params)
print("\nThe size of the vocabulary (which is also the size of the one-hot vecs for words): %d\n\n" % vocab_size)

text_cl.run_code_for_training_for_text_classification_no_gru(model, hidden_size)

import pymsgbox
response = pymsgbox.confirm("Finished training.  Start testing on unseen data?")
if response == "OK": 
    text_cl.run_code_for_testing_text_classification_no_gru(model, hidden_size)


