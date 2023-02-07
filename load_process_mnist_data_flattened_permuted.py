# -*- coding: utf-8 -*-
"""
Created on Mon Apr  4 11:32:59 2022

@author: valentin
"""

import tensorflow as tf
#physical_devices = tf.config.experimental.list_physical_devices('GPU')
#tf.config.experimental.set_memory_growth(physical_devices[0], True)
from tensorflow.keras.optimizers import Adam
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np
import matplotlib.pyplot as plt
from fast_gradient_method import fast_gradient_method
from projected_gradient_method import projected_gradient_method
import time
import ssl
ssl._create_default_https_context = ssl._create_unverified_context


batch_size = 50

num_classes=10
input_shape = 784
(x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()

    
x_train=x_train/255
x_test=x_test/255

# Reserve 10,000 samples for validation.
x_val = x_train[-10000:]
y_val = y_train[-10000:]
x_train = x_train[:-10000]
y_train = y_train[:-10000]

x_train=np.reshape(x_train, (x_train.shape[0],x_train.shape[1]*x_train.shape[1]))
x_val=np.reshape(x_val, (x_val.shape[0],x_val.shape[1]*x_val.shape[1]))
x_test=np.reshape(x_test, (x_test.shape[0],x_test.shape[1]*x_test.shape[1]))

x_train = x_train.astype("float32") 
x_test = x_test.astype("float32") 
x_val = x_val.astype("float32") 

y_train=tf.one_hot(y_train, 10)
y_val=tf.one_hot(y_val, 10)
y_test=tf.one_hot(y_test, 10)
    
permutation1=np.random.permutation(x_train.shape[1])
permutation2=np.random.permutation(x_train.shape[1])
permutation3=np.random.permutation(x_train.shape[1])
permutation4=np.random.permutation(x_train.shape[1])
permutation5=np.random.permutation(x_train.shape[1])
permutation6=np.random.permutation(x_train.shape[1])
permutation7=np.random.permutation(x_train.shape[1])
permutation8=np.random.permutation(x_train.shape[1])
permutation9=np.random.permutation(x_train.shape[1])

x_train_0=x_train
x_train_1=x_train[:,permutation1] 
x_train_2=x_train[:,permutation2] 
x_train_3=x_train[:,permutation3] 
x_train_4=x_train[:,permutation4] 
x_train_5=x_train[:,permutation5] 
x_train_6=x_train[:,permutation6] 
x_train_7=x_train[:,permutation7] 
x_train_8=x_train[:,permutation8] 
x_train_9=x_train[:,permutation9] 


x_val_0=x_val
x_val_1=x_val[:,permutation1] 
x_val_2=x_val[:,permutation2] 
x_val_3=x_val[:,permutation3] 
x_val_4=x_val[:,permutation4] 
x_val_5=x_val[:,permutation5] 
x_val_6=x_val[:,permutation6] 
x_val_7=x_val[:,permutation7] 
x_val_8=x_val[:,permutation8] 
x_val_9=x_val[:,permutation9] 




# Prepare the training dataset.
train_dataset_0 = tf.data.Dataset.from_tensor_slices((x_train_0, y_train))
train_dataset_0 = train_dataset_0.shuffle(buffer_size=1024).batch(batch_size)
# Prepare the validation dataset.
val_dataset_0= tf.data.Dataset.from_tensor_slices((x_val_0, y_val))
val_dataset_0 = val_dataset_0.batch(batch_size)


# Prepare the training dataset.
train_dataset_1 = tf.data.Dataset.from_tensor_slices((x_train_1, y_train))
train_dataset_1 = train_dataset_1.shuffle(buffer_size=1024).batch(batch_size)
# Prepare the validation dataset.
val_dataset_1 = tf.data.Dataset.from_tensor_slices((x_val_1, y_val))
val_dataset_1 = val_dataset_1.batch(batch_size)

# Prepare the training dataset.
train_dataset_2 = tf.data.Dataset.from_tensor_slices((x_train_2, y_train))
train_dataset_2 = train_dataset_2.shuffle(buffer_size=1024).batch(batch_size)
# Prepare the validation dataset.
val_dataset_2 = tf.data.Dataset.from_tensor_slices((x_val_2, y_val))
val_dataset_2 = val_dataset_2.batch(batch_size)

# Prepare the training dataset.
train_dataset_3 = tf.data.Dataset.from_tensor_slices((x_train_3, y_train))
train_dataset_3 = train_dataset_3.shuffle(buffer_size=1024).batch(batch_size)
# Prepare the validation dataset.
val_dataset_3 = tf.data.Dataset.from_tensor_slices((x_val_3, y_val))
val_dataset_3 = val_dataset_3.batch(batch_size)


# Prepare the training dataset.
train_dataset_4 = tf.data.Dataset.from_tensor_slices((x_train_4, y_train))
train_dataset_4 = train_dataset_4.shuffle(buffer_size=1024).batch(batch_size)
# Prepare the validation dataset.
val_dataset_4 = tf.data.Dataset.from_tensor_slices((x_val_4, y_val))
val_dataset_4 = val_dataset_4.batch(batch_size)

# Prepare the training dataset.
train_dataset_5 = tf.data.Dataset.from_tensor_slices((x_train_5, y_train))
train_dataset_5 = train_dataset_5.shuffle(buffer_size=1024).batch(batch_size)
# Prepare the validation dataset.
val_dataset_5 = tf.data.Dataset.from_tensor_slices((x_val_5, y_val))
val_dataset_5 = val_dataset_5.batch(batch_size)

# Prepare the training dataset.
train_dataset_6 = tf.data.Dataset.from_tensor_slices((x_train_6, y_train))
train_dataset_6 = train_dataset_6.shuffle(buffer_size=1024).batch(batch_size)
# Prepare the validation dataset.
val_dataset_6 = tf.data.Dataset.from_tensor_slices((x_val_6, y_val))
val_dataset_6 = val_dataset_6.batch(batch_size)


# Prepare the training dataset.
train_dataset_7 = tf.data.Dataset.from_tensor_slices((x_train_7, y_train))
train_dataset_7 = train_dataset_7.shuffle(buffer_size=1024).batch(batch_size)
# Prepare the validation dataset.
val_dataset_7 = tf.data.Dataset.from_tensor_slices((x_val_7, y_val))
val_dataset_7 = val_dataset_7.batch(batch_size)

# Prepare the training dataset.
train_dataset_8 = tf.data.Dataset.from_tensor_slices((x_train_8, y_train))
train_dataset_8 = train_dataset_8.shuffle(buffer_size=1024).batch(batch_size)
# Prepare the validation dataset.
val_dataset_8 = tf.data.Dataset.from_tensor_slices((x_val_8, y_val))
val_dataset_8 = val_dataset_8.batch(batch_size)


# Prepare the training dataset.
train_dataset_9 = tf.data.Dataset.from_tensor_slices((x_train_9, y_train))
train_dataset_9 = train_dataset_9.shuffle(buffer_size=1024).batch(batch_size)
# Prepare the validation dataset.
val_dataset_9 = tf.data.Dataset.from_tensor_slices((x_val_9, y_val))
val_dataset_9 = val_dataset_9.batch(batch_size)