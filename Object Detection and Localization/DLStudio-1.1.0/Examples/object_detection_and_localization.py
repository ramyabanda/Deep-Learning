#!/usr/bin/env python

##  object_detection_and_localization.py

"""
This script shows how you can use the functionality provided by the inner class
DetectAndLocalize of the DLStudio module for experimenting with object detection and
localization.

Detecting and localizing objects in images is a more difficult problem than just
classifying the objects.  The former requires that your CNN make two different types
of inferences simultaneously, one for classification and the other for localization.
For the localization part, the CNN must carry out what is known as regression. What
that means is that the CNN must output the numerical values for the bounding box that
encloses the object that was detected.  Generating these two types of inferences
requires two different loss functions, one for classification and the other for
regression.

Training a CNN to solve the detection and localization problem requires a dataset
that, in addition to the class labels for the objects, also provides bounding-box
annotations for the objects in the images.  As you see in the code below, this
script uses the PurdueShapes5 dataset for that purpose.
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
                  dataroot = "/home/kak/ImageDatasets/PurdueShapes5/",
                  image_size = [32,32],
                  path_saved_model = "./saved_model",
                  momentum = 0.9,
                  learning_rate = 1e-4,
                  epochs = 2,
                  batch_size = 4,
                  classes = ('rectangle','triangle','disk','oval','star'),
                  debug_train = 1,
                  debug_test = 1,
                  use_gpu = True,
              )


detector = DLStudio.DetectAndLocalize( dl_studio = dls )
dataserver_train = DLStudio.DetectAndLocalize.PurdueShapes5Dataset(
                                   train_or_test = 'train',
                                   dl_studio = dls,
#                                   dataset_file = "PurdueShapes5-20-train.gz", 
                                   dataset_file = "PurdueShapes5-10000-train.gz", 
                                                                      )
dataserver_test = DLStudio.DetectAndLocalize.PurdueShapes5Dataset(
                                   train_or_test = 'test',
                                   dl_studio = dls,
#                                   dataset_file = "PurdueShapes5-20-test.gz"
                                   dataset_file = "PurdueShapes5-1000-test.gz"
                                                                  )
detector.dataserver_train = dataserver_train
detector.dataserver_test = dataserver_test

detector.load_PurdueShapes5_dataset(dataserver_train, dataserver_test)

model = detector.LOADnet2(skip_connections=True, depth=32)

dls.show_network_summary(model)

detector.run_code_for_training_with_CrossEntropy_and_MSE_Losses(model)
#detector.run_code_for_training_with_CrossEntropy_and_BCE_Losses(model)

import pymsgbox
response = pymsgbox.confirm("Finished training.  Start testing on unseen data?")
if response == "OK": 
    detector.run_code_for_testing_detection_and_localization(model)

