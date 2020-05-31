# -*- coding: utf-8 -*-
"""
Created on Sun May 31 16:13:16 2020
binary classification using high-level API: keras
@author: lwang
"""
import time
import numpy as np 
import pandas as pd 
from matplotlib import pyplot as plt
import tensorflow as tf
import tensorflow.keras as keras 
from tensorflow.keras import models,layers,losses,metrics,optimizers
# %matplotlib inline
# %config InlineBackend.figure_format = 'svg'

#%% generate data
n_positive,n_negative = 2000,2000

#postitive samples, smaller circle
r_p = 5.0 + tf.random.truncated_normal([n_positive,1],0.0,1.0)
theta_p = tf.random.uniform([n_positive,1],0.0,2*np.pi) 
Xp = tf.concat([r_p*tf.cos(theta_p),r_p*tf.sin(theta_p)],axis = 1)
Yp = tf.ones_like(r_p)

#negtive samples, bigger circle
r_n = 8.0 + tf.random.truncated_normal([n_negative,1],0.0,1.0)
theta_n = tf.random.uniform([n_negative,1], 0.0, 2*np.pi) 
Xn = tf.concat([r_n*tf.cos(theta_n), r_n*tf.sin(theta_n)],axis = 1)
Yn = tf.zeros_like(r_n)
print(Xn)

#汇总样本
X = tf.concat([Xp,Xn],axis = 0)
Y = tf.concat([Yp,Yn],axis = 0)
print(X.shape)
print(Y.shape)
#样本洗牌
data = tf.concat([X,Y],axis = 1)
data = tf.random.shuffle(data) # shuffle along dimension 0, i.e., samples
X = data[:,:2] # 0-1 columns
Y = data[:,2:] # 2-end columns

#可视化
plt.figure(figsize = (6,6))
plt.scatter(Xp[:,0].numpy(), Xp[:,1].numpy(), c = "r")
plt.scatter(Xn[:,0].numpy(), Xn[:,1].numpy(), c = "g")
plt.legend(["positive","negative"])

#%% prepare data pipline
n= n_positive+n_negative
# 3/4 for training, #20 samples each batch, buffer_size: samples each time
ds_train = tf.data.Dataset.from_tensor_slices((X[0:n*3//4,:],Y[0:n*3//4,:]))\
            .shuffle(buffer_size = 1000)\
            .batch(200)\
            .prefetch(tf.data.experimental.AUTOTUNE).cache()
# 1/4 for dev
ds_valid = tf.data.Dataset.from_tensor_slices((X[n*3//4:,:],Y[n*3//4:,:]))\
            .batch(200)\
            .prefetch(tf.data.experimental.AUTOTUNE)

# test the data pipline
temp_iter = ds_valid.as_numpy_iterator() # number iterator           
(features, labels) = next(temp_iter) # it can run untill end of the iterator

# i=0
# for features, labels in ds_valid:
#     i+=1
#     print(i)
#     print(features.shape)
#     print(labels.shape)


#%% define model class, inherit from models.Model
tf.keras.backend.clear_session()

inputs = keras.Input(shape= (2))
x = layers.Dense(4, activation="relu")(inputs)
x = layers.Dense(8, activation="relu")(x)
outputs = layers.Dense(1, activation="sigmoid")(x)
model = keras.Model(inputs, outputs)

model.summary()

#%% single integers as class lable (Don't convert Y to one-hot class label)
model.compile(
    optimizer= optimizers.Adam(learning_rate=0.01),
    loss=tf.keras.losses.BinaryCrossentropy(),
    metrics=tf.keras.metrics.BinaryAccuracy(name='acc'),
    )

#%% train
N_epoch = 500
start_time = time.time()

# (B)  Train the model using a dataset pipline
history = model.fit(ds_train, epochs= N_epoch, validation_data= ds_valid)

print('elapsed_time:',  time.time() - start_time)

# plot training process
hist_dict = history.history
fig = plt.figure
plt.plot(hist_dict['loss'],label='loss',linestyle='-')
plt.plot(hist_dict['acc'],label='acc',linestyle='-')
plt.plot(hist_dict['val_loss'],label='val_loss',linestyle='--')
plt.plot(hist_dict['val_acc'],label='val_acc',linestyle='--')
plt.legend(loc='best', fontsize='x-large')
plt.grid(True)

#%% result
fig, (ax1,ax2) = plt.subplots(nrows=1,ncols=2,figsize = (12,5))
ax1.scatter(Xp[:,0].numpy(),Xp[:,1].numpy(),c = "r")
ax1.scatter(Xn[:,0].numpy(),Xn[:,1].numpy(),c = "g")
ax1.legend(["positive","negative"]);
ax1.set_title("y_true");

temp_pred = model.predict(X)
Xp_pred = tf.boolean_mask(X, tf.squeeze(temp_pred >=0.5),axis = 0)
Xn_pred = tf.boolean_mask(X, tf.squeeze(temp_pred <0.5),axis = 0)

ax2.scatter(Xp_pred[:,0].numpy(),Xp_pred[:,1].numpy(),c = "r")
ax2.scatter(Xn_pred[:,0].numpy(),Xn_pred[:,1].numpy(),c = "g")
ax2.legend(["positive","negative"]);
ax2.set_title("y_pred")


