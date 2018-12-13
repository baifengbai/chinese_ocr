#-*- coding:utf-8 -*-
#'''
# Created on 18-10-19 下午4:11
#
# @Author: Greg Gao(laygin)
#'''
from keras.applications import VGG16
from keras.models import Model
from keras.layers import Conv2D, GRU, Lambda, Bidirectional, Activation
import tensorflow as tf


def reshape(x):
    b = tf.shape(x)
    x = tf.reshape(x, [b[0] * b[1], b[2], b[3]])
    return x


def reshape2(x):
    x1, x2 = x
    b = tf.shape(x2)
    x = tf.reshape(x1, [b[0], b[1], b[2], 256])
    return x


def reshape3(x):
    b = tf.shape(x)
    x = tf.reshape(x, [b[0], b[1] * b[2] * 10, 2])
    return x


def create_ctpn_model(input=(None, None, 3)):
    base_model = VGG16(weights=None, include_top=False, input_shape=input)
    base_layers = base_model.get_layer('block5_conv3').output

    x = Conv2D(512, (3, 3), strides=(1, 1), padding='same', activation='relu',
               name='rpn_conv1')(base_layers)

    x1 = Lambda(reshape, output_shape=(None, 512))(x)

    x2 = Bidirectional(GRU(128, return_sequences=True), name='blstm')(x1)

    x3 = Lambda(reshape2, output_shape=(None, None, 256))([x2, x])
    x3 = Conv2D(512, (1, 1), padding='same', activation='relu', name='lstm_fc')(x3)

    cls = Conv2D(10 * 2, (1, 1), padding='same', activation='linear', name='rpn_class')(x3)
    regr = Conv2D(10 * 2, (1, 1), padding='same', activation='linear', name='rpn_regress')(x3)

    cls = Lambda(reshape3, output_shape=(None, 2), name='rpn_class_reshape')(cls)
    cls_prob = Activation('softmax', name='rpn_cls_softmax')(cls)

    regr = Lambda(reshape3, output_shape=(None, 2), name='rpn_regress_reshape')(regr)

    infer_mode = Model(base_model.input, [cls, regr, cls_prob])

    return infer_mode

