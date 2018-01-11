# import itertools
# import collections
# import torch
# import numpy as np
import os
import torch
import numpy
import random
import datetime
import argparse

def merge_dictionaries(dict1, dict2):
    for key in dict2:
        if key in dict1 and isinstance(dict1[key], dict) and isinstance(dict2[key], dict):
            merge_dictionaries(dict1[key], dict2[key])
        else:
            dict1[key] = dict2[key]

def update_values(dict_from, dict_to):
    for key, value in dict_from.items():
        if isinstance(value, dict):
            update_values(dict_from[key], dict_to[key])
        elif value is not None:
            dict_to[key] = dict_from[key] 
    return dict_to

# to be able to reproduce exps on reload
def set_random_seed(seed):
    # is pytorch dataloader with multi-threads deterministic ?
    # cudnn may not be deterministic anyway
    torch.manual_seed(seed) # on CPU and GPU
    numpy.random.seed(seed) # useful ? not thread safe
    random.seed(seed) # useful ? thread safe


