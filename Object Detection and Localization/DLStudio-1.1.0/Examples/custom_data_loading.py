#!/usr/bin/env python

##  custom_data_loading.py

"""
This script shows how to use the custom dataloader in the inner class
CustomDataLoading of the DLStudio module.  That custom dataloader is meant
specifically for the PurdueShapes5 dataset that is used in object detection and
localization experiments in DLStudio.
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
                  epochs = 4,
                  batch_size = 4,
                  classes = ('rectangle','triangle','disk','oval','star'),
                  debug_train = 0,
                  debug_test = 1,
                  use_gpu = True,
              )

detector = DLStudio.CustomDataLoading( dl_studio = dls )
dataserver_train = DLStudio.CustomDataLoading.PurdueShapes5Dataset(
                                   train_or_test = 'train',
                                   dl_studio = dls,
                                   dataset_file = "PurdueShapes5-10000-train.gz", 
                                                                      )
dataserver_test = DLStudio.CustomDataLoading.PurdueShapes5Dataset(
                                   train_or_test = 'test',
                                   dl_studio = dls,
                                   dataset_file = "PurdueShapes5-1000-test.gz"
                                                                      )
detector.dataserver_train = dataserver_train
detector.dataserver_test = dataserver_test

detector.load_PurdueShapes5_dataset(dataserver_train, dataserver_test)

model = detector.BMEnet(skip_connections=True, depth=32)

dls.show_network_summary(model)

detector.run_code_for_training_with_custom_loading(model)

detector.run_code_for_testing_with_custom_loading(model)

