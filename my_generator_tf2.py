import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
import collections
from tqdm import tqdm_notebook as tqdm
from datetime import datetime
import math
import cv2
import tensorflow as tf
import keras
import sys
from keras_applications.inception_v3 import InceptionV3, preprocess_input
from sklearn.model_selection import train_test_split, StratifiedKFold, KFold
from sklearn.utils import class_weight, shuffle
import albumentations as album
from one_cycle_lr import OneCycleScheduler
from model import create_model
from keras.optimizers import Adam
from keras.utils import Sequence
#################################
NUM_CLASSES=6


def MyGenerator(list_IDs, labels=None, batch_size=8, img_size=(224, 224), 
                mode='train', augmentation=None, use_labels=True):

    if(mode=='train' or mode=='valid'):
        img_dir = 'data/stage_1_train_3w/'
    else:
        img_dir = 'data/stage_1_test_3w/'

    # Get total number of samples in the data
    n = len(list_IDs)
    nb_batches = int(np.ceil(n/batch_size))
    # Get a numpy array of all the indices of the input data
    indices = np.arange(n)

    # def train_generate():
    while True:
        if(mode=='train'):
            np.random.shuffle(indices)
        for i in range(nb_batches):
            # get the next batch 
            next_batch_indices = indices[i*batch_size:(i+1)*batch_size]
            nb_examples = len(next_batch_indices)

            X = np.empty((nb_examples, img_size[0], img_size[1], 3))
            Y = np.empty((nb_examples, NUM_CLASSES))
            
            for i, ID in enumerate(next_batch_indices):
                image = cv2.imread(img_dir+list_IDs[ID]+".jpg")
                if(not augmentation is None):
                    img_augmented = augmentation(image=image)
                    image = img_augmented['image']
                ########### norm per image ########
                image = image.astype(np.float32)
                mean = np.mean(image.reshape(-1, 3), axis=0)
                std = np.std(image.reshape(-1, 3), axis=0)
                image -= mean
                image /= (std + 0.0000001)
                X[i, :, :, :] = image
                if(mode=='train' or mode=='valid'):
                    label = labels.loc[list_IDs[ID]].values
                    Y[i, :] = label
            
            if(use_labels==True):
                yield X, {'mix2_output': Y, 'mix4_output': Y, 'mix6_output': Y,
                          'mix8_output': Y, 'mix9_output': Y, 'mix10_output': Y}
            else:
                yield X
#############################################################################