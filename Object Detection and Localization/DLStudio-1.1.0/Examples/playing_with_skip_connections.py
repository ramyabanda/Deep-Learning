#!/usr/bin/env python

##  playing_with_skip_connections.py

"""
This script illustrates how to actually use the inner class SkipConnections of
the DLStudio module.

As shown in the calls below, a CNN is constructed by calling on the constructor for
the BMEnet class.

You can easily create a CNN with arbitrary depth just by using the "depth"
constructor option for the BMEnet class.  BMEnet creates a network by using
multiple blocks of SkipBlock.
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
                  dataroot = "/home/kak/ImageDatasets/CIFAR-10/",
                  image_size = [32,32],
                  path_saved_model = "./saved_model",
                  momentum = 0.9,
                  learning_rate = 1e-4,
                  epochs = 6,
                  batch_size = 4,
                  classes = ('plane','car','bird','cat','deer',
                             'dog','frog','horse','ship','truck'),
                  debug_train = 0,
                  debug_test = 0,
                  use_gpu = True,
              )

exp_skip = DLStudio.SkipConnections( dl_studio = dls )

#exp_skip.load_cifar_10_dataset_with_augmentation()
exp_skip.load_cifar_10_dataset()

model = exp_skip.BMEnet(skip_connections=True, depth=32)

dls.show_network_summary(model)

exp_skip.run_code_for_training(model)

exp_skip.run_code_for_testing(model)

