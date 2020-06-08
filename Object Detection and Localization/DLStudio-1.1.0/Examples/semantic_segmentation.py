#!/usr/bin/env python

##  semantic_segmentation.py

"""
This script should be your starting point if you wish to learn how to use the
mUnet neural network for semantic segmentation of images.  As mentioned elsewhere in
the main documentation page, mUnet assigns an output channel to each different type of
object that you wish to segment out from an image. So, given a test image at the
input to the network, all you have to do is to examine each channel at the output for
segmenting out the objects that correspond to that output channel.

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
                  dataroot = "/home/kak/ImageDatasets/PurdueShapes5MultiObject/",
                  image_size = [64,64],
                  path_saved_model = "./saved_model",
                  momentum = 0.9,
                  learning_rate = 1e-4,
                  epochs = 6,
                  batch_size = 4,
                  classes = ('rectangle','triangle','disk','oval','star'),
                  debug_train = 1,
                  debug_test = 1,
                  use_gpu = True,
              )

segmenter = DLStudio.SemanticSegmentation( dl_studio = dls )

dataserver_train = DLStudio.SemanticSegmentation.PurdueShapes5MultiObjectDataset(
                          train_or_test = 'train',
                          dl_studio = dls,
#                          dataset_file = "PurdueShapes5MultiObject-20-train.gz", 
                          dataset_file = "PurdueShapes5MultiObject-10000-train.gz", 
                        )
dataserver_test = DLStudio.SemanticSegmentation.PurdueShapes5MultiObjectDataset(
                          train_or_test = 'test',
                          dl_studio = dls,
#                          dataset_file = "PurdueShapes5MultiObject-20-test.gz"
                          dataset_file = "PurdueShapes5MultiObject-1000-test.gz"
                        )
segmenter.dataserver_train = dataserver_train
segmenter.dataserver_test = dataserver_test

segmenter.load_PurdueShapes5MultiObject_dataset(dataserver_train, dataserver_test)

model = segmenter.mUnet(skip_connections=True, depth=4)
#model = segmenter.mUnet(skip_connections=False, depth=4)

dls.show_network_summary(model)

segmenter.run_code_for_training_for_semantic_segmentation(model)

import pymsgbox
response = pymsgbox.confirm("Finished training.  Start testing on unseen data?")
if response == "OK": 
    segmenter.run_code_for_testing_semantic_segmentation(model)

