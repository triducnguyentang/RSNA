from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import (Activation, Dropout, Flatten, Dense, GlobalMaxPooling2D,
                          BatchNormalization, Input, Conv2D, Multiply, Lambda,
                          Concatenate, GlobalAveragePooling2D, Softmax)
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras import metrics
from tensorflow.keras.models import Model
from tensorflow.keras.applications.inception_v3 import InceptionV3, preprocess_input



function = 'sigmoid'

def create_model(input_shape, n_out=6):
    input_tensor = Input(shape=input_shape)
    base_model = InceptionV3(include_top=False,
                   weights='imagenet',
                   input_tensor=input_tensor)
    # mix2
    mix2_in = base_model.get_layer('mixed2').output
    mix2_maxpool = GlobalMaxPooling2D()(mix2_in)
    mix2_avgpool = GlobalAveragePooling2D()(mix2_in)
    mix2 = Concatenate()([mix2_maxpool,mix2_avgpool])
    mix2 = BatchNormalization()(mix2)
    mix2_output = Dense(n_out, activation=function,
                       name='mix2_output')(mix2)

    # mix4
    mix4_in = base_model.get_layer('mixed4').output
    mix4_maxpool = GlobalMaxPooling2D()(mix4_in)
    mix4_avgpool = GlobalAveragePooling2D()(mix4_in)
    mix4 = Concatenate()([mix4_maxpool,mix4_avgpool])
    mix4 = BatchNormalization()(mix4)
    mix4_output = Dense(n_out, activation=function,
                       name='mix4_output')(mix4)

    # mix6
    mix6_in = base_model.get_layer('mixed6').output
    mix6_maxpool = GlobalMaxPooling2D()(mix6_in)
    mix6_avgpool = GlobalAveragePooling2D()(mix6_in)
    mix6 = Concatenate()([mix6_maxpool,mix6_avgpool])
    mix6 = BatchNormalization()(mix6)
    mix6_output = Dense(n_out, activation=function,
                        name='mix6_output')(mix6)

    # mix8
    mix8_in = base_model.get_layer('mixed8').output
    mix8_maxpool = GlobalMaxPooling2D()(mix8_in)
    mix8_avgpool = GlobalAveragePooling2D()(mix8_in)
    mix8 = Concatenate()([mix8_maxpool,mix8_avgpool])
    mix8 = BatchNormalization()(mix2)
    mix8_output = Dense(n_out, activation=function,
                        name='mix8_output')(mix8)

    # mix9
    mix9_in = base_model.get_layer('mixed9').output
    mix9_maxpool = GlobalMaxPooling2D()(mix9_in)
    mix9_avgpool = GlobalAveragePooling2D()(mix9_in)
    mix9 = Concatenate()([mix9_maxpool,mix9_avgpool])
    mix9 = BatchNormalization()(mix9)
    mix9_output = Dense(n_out, activation=function,
                        name='mix9_output')(mix9)

    # mix10
    mix10_in = base_model.output
    mix10_maxpool = GlobalMaxPooling2D()(mix10_in)
    mix10_avgpool = GlobalAveragePooling2D()(mix10_in)
    mix10_concat = Concatenate()([mix10_maxpool,mix10_avgpool])
    mix10 = Dropout(0.3)(mix10_concat)
    mix10 = Dense(512, activation='relu')(mix10)
    mix10 = Dropout(0.3)(mix10)
    mix10_output = Dense(n_out, activation=function,
                         name='mix10_output')(mix10)
    
    model = Model(input_tensor, [mix2_output, mix4_output, mix6_output,
                                 mix8_output, mix9_output, mix10_output])
    
    return model