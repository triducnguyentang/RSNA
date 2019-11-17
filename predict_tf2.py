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
from sklearn.model_selection import train_test_split, StratifiedKFold, KFold
from sklearn.utils import class_weight, shuffle
import albumentations as album
from model_tf2 import create_model
from my_generator_tf2 import MyGenerator


SIZE=448
NUM_CLASSES=6
BATCH_SIZE=16*4*2
NUM_FOLD=5


def read_testset(filename="data/stage_1_sample_submission.csv"):
    df = pd.read_csv(filename)
    df["Image"] = df["ID"].str.slice(stop=12)
    df["Diagnosis"] = df["ID"].str.slice(start=13)
    
    df = df.loc[:, ["Label", "Diagnosis", "Image"]]
    df = df.set_index(['Image', 'Diagnosis']).unstack(level=-1)
    
    return df

test_df = read_testset()
print(test_df.shape, test_df.head())


# Augmentation
albumentations_test = album.Compose([
    album.Resize(SIZE,SIZE)
], p=1)

albumentations_flip = album.Compose([
    album.HorizontalFlip(p=1),
    album.Resize(SIZE,SIZE)
], p=1)

# generator
test_generator = MyGenerator(test_df.index, test_df, batch_size=BATCH_SIZE,
    img_size=(SIZE,SIZE), mode='test', augmentation=albumentations_test, use_labels=False)
test_gen_dataset = tf.data.Dataset.from_generator(
        lambda:test_generator,
        output_types=('float32'),
        output_shapes=(tf.TensorShape((None, SIZE, SIZE, 3))) )

test_flip = MyGenerator(test_df.index, test_df, batch_size=BATCH_SIZE,
    img_size=(SIZE,SIZE), mode='test', augmentation=albumentations_flip, use_labels=False)
test_flip_gen_dataset = tf.data.Dataset.from_generator(
        lambda:test_flip,
        output_types=('float32'),
        output_shapes=(tf.TensorShape((None, SIZE, SIZE, 3))) )


y_preds = 0

for fold in range(NUM_FOLD):

    print('****** FOLD: ', fold)

    strategy = tf.distribute.MirroredStrategy()
    with strategy.scope():
        model = create_model(
            input_shape=(SIZE,SIZE,3), 
            n_out=NUM_CLASSES)
        model.load_weights('working/InceptionV3_3w_{}_fold{}.h5'.format(SIZE, fold))

        y_preds_non = model.predict(test_gen_dataset,
            steps=math.ceil(float(len(test_df)) / float(BATCH_SIZE)),
            workers=24, max_queue_size=10,
            use_multiprocessing=True,
            verbose=1)
        tmp = (y_preds_non[2] + y_preds_non[4] + y_preds_non[5])/3.
        y_preds_non = np.array(tmp)

        y_preds_flip = model.predict(test_flip_gen_dataset,
            steps=math.ceil(float(len(test_df)) / float(BATCH_SIZE)),
            workers=24, max_queue_size=10,
            use_multiprocessing=True,
            verbose=1)
        tmp = (y_preds_flip[2] + y_preds_flip[4] + y_preds_flip[5])/3.
        y_preds_flip = np.array(tmp)

        y_preds += (y_preds_non + y_preds_flip)/2

y_preds /= NUM_FOLD

test_df.iloc[:, :] = y_preds
test_df = test_df.stack().reset_index()
test_df.insert(loc=0, column='ID', value=test_df['Image'].astype(str) + "_" + test_df['Diagnosis'])
test_df = test_df.drop(["Image", "Diagnosis"], axis=1)
test_df.to_csv('InceptionV3.csv', index=False)