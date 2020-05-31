# -*- coding: utf-8 -*-
"""
Created on Sun May 31 10:45:16 2020
binary classification using medium-level API: write a subclass of models.Model
@author: lwang
"""

import numpy as np 
import pandas as pd 
from matplotlib import pyplot as plt
import tensorflow as tf
from tensorflow.keras import models,layers,losses,metrics,optimizers
# %matplotlib inline
# %config InlineBackend.figure_format = 'svg'

#%% 打印时间分割线
# @tf.function
def printbar():
    today_ts = tf.timestamp()%(24*60*60)

    hour = tf.cast(today_ts//3600+8,tf.int32)%tf.constant(24)
    minite = tf.cast((today_ts%3600)//60,tf.int32)
    second = tf.cast(tf.floor(today_ts%60),tf.int32)
    
    def timeformat(m):
        if tf.strings.length(tf.strings.format("{}",m))==1:
            return(tf.strings.format("0{}",m))
        else:
            return(tf.strings.format("{}",m))
    
    timestring = tf.strings.join([timeformat(hour),timeformat(minite),
                timeformat(second)],separator = ":")
    tf.print("=========="*8+timestring)
    
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
class DNNModel(models.Model):
    def __init__(self):
        super(DNNModel, self).__init__() # same as super class: DNNModel
        
    def build(self,input_shape):
        self.dense1 = layers.Dense(4,activation = "relu",name = "dense1") 
        self.dense2 = layers.Dense(8,activation = "relu",name = "dense2")
        self.dense3 = layers.Dense(1,activation = "sigmoid",name = "dense3")
        super(DNNModel,self).build(input_shape)
 
    # 正向传播
    @tf.function(input_signature=[tf.TensorSpec(shape = [None,2], dtype = tf.float32)])  
    def call(self,x):
        x = self.dense1(x)
        x = self.dense2(x)
        y = self.dense3(x)
        return y

model = DNNModel()
model.build(input_shape =(None,2))

model.summary()

#%% define for-loop train
optimizer = optimizers.Adam(learning_rate=0.01)
loss_func = tf.keras.losses.BinaryCrossentropy()

train_loss = tf.keras.metrics.Mean(name='train_loss')
train_metric = tf.keras.metrics.BinaryAccuracy(name='train_accuracy')

valid_loss = tf.keras.metrics.Mean(name='valid_loss')
valid_metric = tf.keras.metrics.BinaryAccuracy(name='valid_accuracy')

@tf.function #8s/100 epochs, 2 mins without it
def train_step(model, features, labels):
    with tf.GradientTape() as tape:
        predictions = model(features)
        loss = loss_func(labels, predictions)
    grads = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(grads, model.trainable_variables))

    train_loss.update_state(loss)
    train_metric.update_state(labels, predictions)

@tf.function # same with above
def valid_step(model, features, labels):
    predictions = model(features)
    batch_loss = loss_func(labels, predictions)
    valid_loss.update_state(batch_loss)
    valid_metric.update_state(labels, predictions)
    
    
def train_model(model,ds_train,ds_valid,epochs):
    for epoch in tf.range(1,epochs+1):
        for features, labels in ds_train:
            train_step(model,features,labels)

        for features, labels in ds_valid:
            valid_step(model,features,labels)

        logs = 'Epoch={},Loss:{},Accuracy:{},Valid Loss:{},Valid Accuracy:{}'
        
        if  epoch%100 ==0:
            printbar()
            tf.print(tf.strings.format(logs,
            (epoch,train_loss.result(),train_metric.result(),valid_loss.result(),valid_metric.result())))
        
        train_loss.reset_states()
        valid_loss.reset_states()
        train_metric.reset_states()
        valid_metric.reset_states()
 

train_model(model,ds_train,ds_valid, 500)

#%% result
fig, (ax1,ax2) = plt.subplots(nrows=1,ncols=2,figsize = (12,5))
ax1.scatter(Xp[:,0].numpy(),Xp[:,1].numpy(),c = "r")
ax1.scatter(Xn[:,0].numpy(),Xn[:,1].numpy(),c = "g")
ax1.legend(["positive","negative"]);
ax1.set_title("y_true");

Xp_pred = tf.boolean_mask(X, tf.squeeze(model(X)>=0.5),axis = 0)
Xn_pred = tf.boolean_mask(X, tf.squeeze(model(X)<0.5),axis = 0)

ax2.scatter(Xp_pred[:,0].numpy(),Xp_pred[:,1].numpy(),c = "r")
ax2.scatter(Xn_pred[:,0].numpy(),Xn_pred[:,1].numpy(),c = "g")
ax2.legend(["positive","negative"]);
ax2.set_title("y_pred")


