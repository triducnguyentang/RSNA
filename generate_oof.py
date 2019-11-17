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
import sys
from sklearn.utils import class_weight, shuffle
import albumentations as album
from model_tf2 import create_model
from my_generator_tf2 import MyGenerator


SIZE=448
NUM_CLASSES=6
BATCH_SIZE=16*4*2
NUM_FOLD=5


# Augmentation
albumentations_test = album.Compose([
    album.Resize(SIZE,SIZE)
], p=1)

albumentations_flip = album.Compose([
    album.HorizontalFlip(p=1),
    album.Resize(SIZE,SIZE)
], p=1)


for fold in range(NUM_FOLD):

    print('***** fold: ', )
    valid_df = pd.read_csv('csv/valid_{}.csv'.format(fold))
    print(valid_df.shape, valid_df.head())

    valid_generator = MyGenerator(valid_df['sop_instance_uid'],
        valid_df.drop(['patient_id'], axis=1).set_index('sop_instance_uid'), batch_size=BATCH_SIZE,
        img_size=(SIZE,SIZE), mode='valid', augmentation=albumentations_test, use_labels=False)
    valid_gen_dataset = tf.data.Dataset.from_generator(
            lambda:valid_generator,
            output_types=('float32'),
            output_shapes=(tf.TensorShape((None, SIZE, SIZE, 3))) )


    valid_flip = MyGenerator(valid_df['sop_instance_uid'],
        valid_df.drop(['patient_id'], axis=1).set_index('sop_instance_uid'), batch_size=BATCH_SIZE,
        img_size=(SIZE,SIZE), mode='valid', augmentation=albumentations_flip, use_labels=False)
    valid_flip_gen_dataset = tf.data.Dataset.from_generator(
            lambda:valid_flip,
            output_types=('float32'),
            output_shapes=(tf.TensorShape((None, SIZE, SIZE, 3))) )

    strategy = tf.distribute.MirroredStrategy()
    with strategy.scope():
        model = create_model(
            input_shape=(SIZE,SIZE,3), 
            n_out=NUM_CLASSES)
        model.load_weights('working/InceptionV3_3w_{}_fold{}.h5'.format(SIZE, fold))

        y_preds_non = model.predict(valid_gen_dataset,
            steps=math.ceil(float(len(valid_df)) / float(BATCH_SIZE)),
            workers=24, max_queue_size=10,
            use_multiprocessing=True,
            verbose=1)
        tmp = (y_preds_non[2] + y_preds_non[4] + y_preds_non[5])/3.
        y_preds_non = np.array(tmp)

        y_preds_flip = model.predict(valid_flip_gen_dataset,
            steps=math.ceil(float(len(valid_df)) / float(BATCH_SIZE)),
            workers=24, max_queue_size=10,
            use_multiprocessing=True,
            verbose=1)
        tmp = (y_preds_flip[2] + y_preds_flip[4] + y_preds_flip[5])/3.
        y_preds_flip = np.array(tmp)

        y_preds = (y_preds_non + y_preds_flip)/2

    for i in range(6):
        valid_df.iloc[:, i+1] = y_preds[:, i]
    valid_df.to_csv('working/3w_{}_tta_fold{}_valid.csv'.format(SIZE, fold), index=False)