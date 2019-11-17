import numpy as np
import pandas as pd
import pydicom
import os
import matplotlib.pyplot as plt
import collections
from tqdm import tqdm_notebook as tqdm
from datetime import datetime
import math
import cv2
import tensorflow as tf
import tensorflow.keras as keras
import sys
from tensorflow.keras.applications.inception_v3 import InceptionV3, preprocess_input
from sklearn.model_selection import train_test_split, StratifiedKFold, KFold
from sklearn.utils import class_weight, shuffle
import albumentations as album
from one_cycle_lr_tf2 import OneCycleScheduler
from model_tf2 import create_model
from tensorflow.keras.optimizers import Adam
from tensorflow.python.keras.utils.data_utils import Sequence
from my_generator_tf2 import MyGenerator
from tensorflow.keras.losses import binary_crossentropy
import random
from tensorflow.keras.callbacks import (ModelCheckpoint, LearningRateScheduler,
                             EarlyStopping, ReduceLROnPlateau,CSVLogger)

SIZE=448
BATCH_SIZE=12*4*2
NUM_CLASSES=6
NUM_FOLD=5

######################################################################
# under-sampling method
def get_balance_set(df):
    patients = set(df["patient_id"].unique())
    patients_pos = set(df[df["any"] == 1]["patient_id"].unique())
    patients_neg = patients - patients_pos
    patients_neg_balance = random.sample(patients_neg, len(patients_pos))
    patients_balance = patients_pos.union(patients_neg_balance)
    
    print(len(patients), len(patients_pos), len(patients), len(patients_balance))
    
    return df[df["patient_id"].isin(patients_balance)].reset_index()


# Augmentation
albumentations_train = album.Compose([
    album.Cutout(num_holes=8),
    album.Resize(SIZE,SIZE),
    album.HorizontalFlip(),
    album.RandomRotate90(),
    album.RandomBrightnessContrast(),
    album.GridDistortion(distort_limit=0.2),
], p=1)

albumentations_valid = album.Compose([
    album.Resize(SIZE,SIZE)
], p=1)
######################################################################

# define weighted loss for keras
def _weighted_log_loss(y_true, y_pred):
    
    class_weights = np.array([2, 1, 1, 1, 1, 1])

    y_pred = tf.keras.backend.clip(y_pred, tf.keras.backend.epsilon(), 1.0-tf.keras.backend.epsilon())
    out = -(         y_true  * tf.keras.backend.log(      y_pred) * class_weights
            + (1.0 - y_true) * tf.keras.backend.log(1.0 - y_pred) * class_weights)
    
    return tf.keras.backend.mean(out, axis=-1)

# parameter for Deep Supervision
losses = {
    'mix2_output': _weighted_log_loss,
    'mix4_output': _weighted_log_loss,
    'mix6_output': _weighted_log_loss,
    'mix8_output': _weighted_log_loss,
    'mix9_output': _weighted_log_loss,
    'mix10_output': _weighted_log_loss,
}

lossWeights = {'mix2_output': 0.05,
        'mix4_output': 0.05,
        'mix6_output': 0.1,
        'mix8_output': 0.15,
        'mix9_output': 0.2,
        'mix10_output': 1.0,  
      }
########################################################################
for fold in range(NUM_FOLD):

    # load dataset
    df_train = pd.read_csv('csv/train_{}.csv'.format(fold))
    df_valid = pd.read_csv('csv/valid_{}.csv'.format(fold))

    # under-sampling
    df_train = get_balance_set(df_train).drop(['index'],axis=1)

    # create generator
    train_generator = MyGenerator(df_train['sop_instance_uid'],
        df_train.drop(['patient_id'], axis=1).set_index('sop_instance_uid'), batch_size=BATCH_SIZE,
    	img_size=(SIZE,SIZE), mode='train', augmentation=albumentations_train)
    valid_generator = MyGenerator(df_valid['sop_instance_uid'],
        df_valid.drop(['patient_id'], axis=1).set_index('sop_instance_uid'), batch_size=BATCH_SIZE,
    	img_size=(SIZE,SIZE), mode='valid', augmentation=albumentations_valid)

    
    train_gen_dataset = tf.data.Dataset.from_generator(
            lambda:train_generator,
            output_types=('float32', {'mix2_output': 'float32', 'mix4_output': 'float32',
                                      'mix6_output': 'float32', 'mix8_output': 'float32',
                                      'mix9_output': 'float32', 'mix10_output': 'float32'}),
            output_shapes=(tf.TensorShape((None, SIZE, SIZE, 3)),
                {'mix2_output': tf.TensorShape((None, 6)), 'mix4_output': tf.TensorShape((None, 6)),
                 'mix6_output': tf.TensorShape((None, 6)), 'mix8_output': tf.TensorShape((None, 6)),
                 'mix9_output': tf.TensorShape((None, 6)), 'mix10_output': tf.TensorShape((None, 6))}) )
    valid_gen_dataset = tf.data.Dataset.from_generator(
            lambda:valid_generator,
            output_types=('float32', {'mix2_output': 'float32', 'mix4_output': 'float32',
                                      'mix6_output': 'float32', 'mix8_output': 'float32',
                                      'mix9_output': 'float32', 'mix10_output': 'float32'}),
            output_shapes=(tf.TensorShape((None, SIZE, SIZE, 3)),
                {'mix2_output': tf.TensorShape((None, 6)), 'mix4_output': tf.TensorShape((None, 6)),
                 'mix6_output': tf.TensorShape((None, 6)), 'mix8_output': tf.TensorShape((None, 6)),
                 'mix9_output': tf.TensorShape((None, 6)), 'mix10_output': tf.TensorShape((None, 6))}) )
    ###################################################################
    strategy = tf.distribute.MirroredStrategy()
    with strategy.scope():
        model = create_model(
                input_shape=(SIZE,SIZE,3), 
                n_out=NUM_CLASSES)
        
        # freeze the backbone
        for layer in model.layers[:-38]:
            layer.trainable=False

        model.compile(optimizer=Adam(lr=1e-2, decay=1e-3),
            loss=losses, loss_weights=lossWeights)
        model.summary()

        model.fit(train_gen_dataset,
                steps_per_epoch=math.ceil(float(len(df_train)) / float(BATCH_SIZE)),
                epochs=2, workers=16, max_queue_size=10,
                use_multiprocessing=True,
                callbacks=[OneCycleScheduler(lr_max=1e-2, lr_min=1e-3, n_data_points=len(df_train),
                                     epochs=2, batch_size=BATCH_SIZE, verbose=0)],
                verbose=1)

        # unfreeze, train all the layers
        for layer in model.layers:
            layer.trainable=True

        weightpath = "working/InceptionV3_3w_448_fold{}.h5".format(fold)
        logpath = 'working/log_448_fold{}.csv'.format(fold)
        callbacks=[
                   ModelCheckpoint(weightpath, monitor='val_loss', verbose=1,
                    save_best_only=True, mode='min', save_weights_only=True),
                   CSVLogger(filename=logpath,separator=',', append=True),
                   ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=1, 
                                       verbose=1, mode='min', epsilon=0.0001),
                   EarlyStopping(monitor="val_loss", mode="min", patience=3)]
            
        model.compile(optimizer=Adam(lr=1e-3, decay=1e-3),
            loss=losses)

        model.fit(train_gen_dataset,
                steps_per_epoch=math.ceil(float(len(df_train)) / float(BATCH_SIZE)),
                validation_data=valid_gen_dataset,
                validation_steps=math.ceil(float(len(df_valid)) / float(BATCH_SIZE)),
                epochs=10, callbacks=callbacks,
                workers=16, max_queue_size=10,
                use_multiprocessing=True,
                verbose=1)