#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
DenseNet model for Cifar10 classification.
"""

import os

import numpy as np
import pickle

from keras.optimizers import SGD
from keras.datasets import cifar10
from keras.utils import to_categorical
from keras.callbacks import LearningRateScheduler, TensorBoard, ModelCheckpoint
import keras.backend as K

import utils
from DenseNet import DenseNet


# Set default data_format
K.set_image_data_format('channels_last')

# Define useful paths
results_name = 'DenseNet-BC_cifar10_L100_k12_32_FIXED'

results_path = os.path.join(os.getcwd(), 'results', results_name)
tb_logs = os.path.join(os.getcwd(), 'logs', results_name)

try:
    os.makedirs(results_path)
except:
    pass

try:
    os.makedirs(tb_logs)
except:
    pass

# Load data
(x_train, y_train), (x_test, y_test) = cifar10.load_data()

num_classes = 10
y_train = to_categorical(y_train, num_classes)
y_test = to_categorical(y_test, num_classes)

# Change mode to channels_last
if x_train.shape[3] != 3:
    x_train = np.swapaxes(x_train, 1, 3)
    x_test = np.swapaxes(x_test, 1, 3)

# Preprocess the images
channel_means = np.mean(x_train, axis=(0,1,2))
channel_stds = np.std(x_train, axis=(0,1,2))

x_train = (x_train - channel_means) / channel_stds
x_test = (x_test - channel_means) / channel_stds

# Create the model
batch_size = 64
epochs = 300

model = DenseNet(growth_rate=12, blocks=[32,32,32], first_num_channels=2*12,
                 dropout_p=0.2, bottleneck=4*12, compression=0.5,
                 input_shape=(32,32,3), first_conv_pool=False,
                 weight_decay=1e-4, data_format='channels_last',
                 num_classes=num_classes)

with open(os.path.join(results_path, 'model_summary'), 'w') as f:
    model.summary(print_fn=lambda x: f.write(x + '\n'))

# Train the model from scratch
lr = 0.1


def schedule_fn(epoch_i):
    if epoch_i >= epochs * 0.75:
        print('learning rate = ', lr/100)
        return lr / 100
    elif epoch_i >= epochs * 0.5:
        print('learning rate = ', lr/10)
        return lr / 10
    else:
        print('learning rate = ', lr)
        return lr


lr_scheduler = LearningRateScheduler(schedule_fn)
tb = TensorBoard(log_dir=tb_logs, batch_size=batch_size)
model_chp = ModelCheckpoint(os.path.join(results_path, 'best_weights'),
                            monitor='val_loss',
                            save_best_only=True, save_weights_only=True)
try:
    os.makedirs(os.path.join(os.getcwd(), 'logs'))
except:
    pass

sgd = SGD(lr=lr, momentum=0.9, nesterov=True)
model.compile(optimizer=sgd, loss='categorical_crossentropy',
              metrics=['accuracy'])

history = model.fit(x=x_train, y=y_train, batch_size=batch_size, epochs=epochs,
                    callbacks=[lr_scheduler, tb, model_chp],
                    validation_data=(x_test, y_test))

# Save and plot history
with open(os.path.join(results_path, 'cifar10_history'), 'wb') as f:
    pickle.dump(history.history, f)

utils.plot_accuracy(history)
utils.plot_loss(history)
