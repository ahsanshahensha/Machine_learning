# -*- coding: utf-8 -*-
"""
Created on Mon Jul 17 19:50:52 2017

@author: ahsan
"""
import tensorflow as tf
import numpy as np
import pylab
#matplotlib inline

'''
linear regression 

y = 0.3 *x +0.1+noise

find weight (0.3)
and bias (0.1)

'''
x_data = np.random.rand(100).astype(np.float32)
noise = np.random.normal(scale=0.01,size=len(x_data))
y_data = 0.3 * x_data+0.1+noise
pylab.plot(x_data,y_data,'r.')
#%%
W = tf.Variable(tf.random_uniform([1],0.0,1.0),name='Weights')
b = tf.Variable(np.zeros([1]).astype(np.float32),name='bias')
y= W*x_data+b
print(W,b)
#%%
loss = tf.reduce_mean(tf.square(y-y_data)) #mean square error
optimizer = tf.train.GradientDescentOptimizer(0.5) #learning rate 0.5
train = optimizer.minimize(loss)#minimize loss
init=tf.global_variables_initializer

#%%
#print(tf.get_default_graph().as_graph_def())

#%%
sess = tf.Session()
sess.run(y)
for i in range(201):
    sess.run(train)#train works on loss; loss needs y; y needs W and b; so all executed
    print(i+sess.run([W,b]))
