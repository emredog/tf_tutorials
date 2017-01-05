# small modifications are made by emredog 
#
#Copyright 2015 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""A very simple MNIST classifier on ConvNet.

See extensive documentation at
https://www.tensorflow.org/tutorials/mnist/pros/
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import sys

from tensorflow.examples.tutorials.mnist import input_data

import tensorflow as tf

data_dir = 'MNIST_data/'

input_img_w = 28
input_img_h = 28
input_img_channels = 1
num_classes = 10
learning_rate = 1e-4


# WEIGHT INIT:
def weight_variable(shape):
  initial = tf.truncated_normal(shape, stddev=0.1)
  return tf.Variable(initial)

def bias_variable(shape):
  initial = tf.constant(0.1, shape=shape)
  return tf.Variable(initial)


# CONVOLUTIONS:
# see http://stackoverflow.com/questions/37674306/what-is-the-difference-between-same-and-valid-padding-in-tf-nn-max-pool-of-t
# about padding methods
def conv2d(x, W):
  return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME') # zero padding, stride=1

def max_pool_2x2(x):
  return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],
                        strides=[1, 2, 2, 1], padding='SAME') #padding with -inf, stride 2


# Import data
mnist = input_data.read_data_sets(data_dir, one_hot=True)  

# x_image = tf.placeholder(tf.float32, shape=[None,input_img_w,input_img_h,input_img_channels]) 
# this does not work due to data provided by mnist.train.next_batch(..), which returns a batch of dim (n,784)  
x = tf.placeholder(tf.float32, [None, input_img_w*input_img_h]) # placeholder for input data
x_image = tf.reshape(x, [-1,input_img_w,input_img_h,1])
y_ = tf.placeholder(tf.float32, [None, num_classes]) # placeholder for ground truth labels


# First Convolutional Layer: Convolution(with RELU) - Max pooling
W_conv1 = weight_variable([5, 5, 1, 32]) # 32 filters of 5x5, depth:1 (since input is grayscale)
b_conv1 = bias_variable([32]) # 32 bias for 32 filters

h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1) # operation for conv1
h_pool1 = max_pool_2x2(h_conv1) # operation for maxpooling (will reduce the size by factor 2 --> 14x14)


# Second Convolutional Layer: Convolution (with RELU) - Max pooling
W_conv2 = weight_variable([5, 5, 32, 64]) # 64 filters of size 5x5, depth:32 (since the output of h_pool1 has 32 slices)
b_conv2 = bias_variable([64]) # 64 biases for 64 filters

h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2) # operation for conv2
h_pool2 = max_pool_2x2(h_conv2) # operation of max pooling (will reduce the size by factor 2 --> 7x7)


# Densely (Fully) Connected Layer
W_fc1 = weight_variable([7 * 7 * 64, 1024]) # 1024 neurons that are connected to every bit of input data for this layer (64 featmaps of size 7x7)
b_fc1 = bias_variable([1024]) # a bias for each neuron

h_pool2_flat = tf.reshape(h_pool2, [-1, 7*7*64]) # flatten the latest output (h_pool2) to a 1D feat vector
h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1) # do a classical Wx+b with a RELU


# Dropout
keep_prob = tf.placeholder(tf.float32) # a scalar for dropout probability (prob. of a neuron to be dropped out)
h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob) # operation for dropout

# Final layer (Readout?)
W_fc2 = weight_variable([1024, num_classes]) # 1024 to 10 
b_fc2 = bias_variable([num_classes]) 

y_conv = tf.matmul(h_fc1_drop, W_fc2) + b_fc2 # final output of one-hot vectors


# loss function
cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(y_conv, y_))

# backpropagation with Adam
train_step = tf.train.AdamOptimizer(learning_rate).minimize(cross_entropy) # optimize with Adam

# define correct predictions and accuracy
correct_prediction = tf.equal(tf.argmax(y_conv,1), tf.argmax(y_,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

# start session & init variables
init = tf.global_variables_initializer()
sess = tf.Session()
sess.run(init)

# do the training and evaluation
for i in range(20000):
  batch = mnist.train.next_batch(50)
  if i%100 == 0:
    train_accuracy = accuracy.eval(session=sess, feed_dict={
      x:batch[0], y_: batch[1], keep_prob: 1.0})
    print("step %d, training accuracy %g"%(i, train_accuracy))
  train_step.run(session=sess, feed_dict={x: batch[0], y_: batch[1], keep_prob: 0.5})


print("test accuracy %g"%accuracy.eval(session=sess, feed_dict={
    x: mnist.test.images, y_: mnist.test.labels, keep_prob: 1.0}))



# if __name__ == '__main__':
#   parser = argparse.ArgumentParser()
#   parser.add_argument('--data_dir', type=str, default='/tmp/tensorflow/mnist/input_data',
#                       help='Directory for storing input data')
#   FLAGS, unparsed = parser.parse_known_args()
#   tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)
