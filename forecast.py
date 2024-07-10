import numpy as np
import random
import os

import torch
torch.use_deterministic_algorithms(True)  # to help make code deterministic
torch.backends.cudnn.benchmark = False  # to help make code deterministic
import torch.nn as nn

np.random.seed(0)  # to help make code deterministic
torch.manual_seed(0)  # to help make code deterministic
random.seed(0)  # to help make code deterministic

import pickle

from UDA_pytorch_utils import UDA_pytorch_classifier_fit, \
    UDA_pytorch_classifier_predict, \
    UDA_compute_accuracy, UDA_get_rnn_last_time_step_outputs

def number_recog(image_array):
    print(image_array.shape)
    input = torch.tensor(np.array([image_array]), dtype=torch.float32)

    deep_convnet = torch.load("deep_convnet.pt")

    number_pred = UDA_pytorch_classifier_predict(deep_convnet,
                               [input],
                               rnn=False).numpy()[0]
    
    return "Your number is: {}".format(number_pred)
    




