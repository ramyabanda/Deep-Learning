#!/usr/bin/env python

##  playing_with_reconfig.py

"""
Shows how you can specify a convolution network with a configuration
string.  The DLStudio module parses the string constructs the network.
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

convo_layers_config = "1x[128,3,3,1]-MaxPool(2) 1x[16,5,5,1]-MaxPool(2)"

##  In the following specification, the entry '-1' for the
##  first element is intentional.  That causes the DLStudio
##  module to set the size of of the input to the fully
##  connected layer to the last activation volume of the
##  convolutional part of the network.
fc_layers_config = [-1,1024,10]

dls = DLStudio(
                  dataroot = "/home/kak/ImageDatasets/CIFAR-10/",
                  image_size = [32,32],
                  convo_layers_config = convo_layers_config,
                  fc_layers_config = fc_layers_config,
                  path_saved_model = "./saved_model",
                  momentum = 0.9,
                  learning_rate = 1e-3,
                  epochs = 2,
                  batch_size = 4,
                  classes = ('plane','car','bird','cat','deer',
                             'dog','frog','horse','ship','truck'),
                  use_gpu = True,
                  debug_train = 0,
                  debug_test = 1,
              )

configs_for_all_convo_layers = dls.parse_config_string_for_convo_layers()

convo_layers = dls.build_convo_layers( configs_for_all_convo_layers )
fc_layers = dls.build_fc_layers()
model = dls.Net(convo_layers, fc_layers)
dls.show_network_summary(model)
dls.load_cifar_10_dataset()
dls.run_code_for_training(model)
dls.run_code_for_testing(model)


