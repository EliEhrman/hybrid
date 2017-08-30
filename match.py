"""
This module is about the simplest example I managed to create of a neural network in Tensorflow

Its main purpose is for use as a starting point for more ambitious experiments. The goal being that whenever
I create something I want to know that at inception it already can learn a simple function. Then as each stage is
added I want to know that I haen't broken the system.

The data created is a simple vector for which either:
	a. The second half of the vector is an identical copy of the first half
	b. All the numbers are random with probably no one matching another
A truth array holds the information of whether the vactor is as in (a), marked 1, 0
or as in (b) and marked (0, 1)

I created just two hidden fully connected layers that end in two nodes. The input is a batch of
vectors as described above. The output is passed to softmax and compared to truth.
Argmax is used to create an accuracy score.
There is no holdout for validation because the dataset is huge relative to the number of parameters.

Basic support for Tensorboard is included

And oh, yes, you should get to 100% accuracy

"""
from __future__ import print_function
# import csv
import numpy as np
import tensorflow as tf
# import random
# import os
# import math
import collections


num_vals = 40
num_samples = 100000
batch_size = 1000
core_dim = 40
num_steps = 10000000
learn_rate = 5.0


data = np.ndarray([num_samples, num_vals], dtype=np.float32)
truth = np.ndarray([num_samples, 2], dtype=np.float32)
for isamp in range(num_samples / 2):
	data[isamp*2, :num_vals/2]=np.random.rand(num_vals/2)
	data[isamp*2, num_vals/2:]=data[isamp*2, :num_vals/2]
	data[isamp*2+1, :]=np.random.rand(num_vals)
	truth[isamp * 2] = [1.0, 0.0]
	truth[isamp * 2 + 1] = [0.0, 1.0]

t_data = tf.constant(data)
t_truth = tf.constant(truth)
t_index = tf.Variable(tf.random_uniform([batch_size], 0, num_samples - 1, dtype=tf.int32), trainable=False)

with tf.name_scope('nn'):
	wfc1 = tf.Variable(tf.random_normal([num_vals, core_dim], 0, 1e-3, dtype=tf.float32), name='t_wfc1')
	bfc1 = tf.Variable(tf.random_normal([core_dim], 0, 1e-3, dtype=tf.float32), name='t_bfc1')
	wfc2 = tf.Variable(tf.random_normal([core_dim, 2], 0, 1e-3, dtype=tf.float32), name='t_wfc2')
	bfc2 = tf.Variable(tf.random_normal([2], 0, 1e-3, dtype=tf.float32), name='t_bfc2')

t_index_set_op = tf.assign(t_index, tf.random_uniform([batch_size], 0, num_samples - 1, dtype=tf.int32))
t_batch = tf.gather(t_data, t_index, name='gather_data')
t_batch_truth = tf.gather(t_truth, t_index, name='gather_truth')

fc1 = tf.sigmoid(tf.matmul(t_batch, wfc1, name='matmul_fc1') + bfc1, name='sig_fc1')
pred = tf.sigmoid(tf.matmul(fc1, wfc2, name='matmul_fc2') + bfc2, name='sig_fc2')

t_err = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=pred,
															   labels=t_batch_truth), name='x_entropy')
train1 = tf.train.GradientDescentOptimizer(learning_rate=learn_rate, name='GDO').minimize(t_err, name='GDOmin')
tf.summary.scalar('Error', t_err)

t_correct = tf.equal(tf.argmax(pred, 1, name='am_pred'), tf.argmax(t_batch_truth, 1, name='am_truth'), name='eq_correct')
t_acc = tf.reduce_mean(tf.cast(t_correct, tf.float32), name='acc_mean')
tf.summary.scalar('Accuracy', t_acc)

"""
The argument tp Session in the following produces an output that shows to which device each and every variable
is allocated. On my machine with a cpu and gpu almost everything is allocated to gpu:0
Accuracy and Error are not. (perhaps because of the summary)
"""
sess = tf.Session(config=tf.ConfigProto(
    log_device_placement=True))
merged = tf.summary.merge_all()
summaries_dir = '/tmp'
train_writer = tf.summary.FileWriter(summaries_dir + '/train',
                                      sess.graph)
# test_writer = tf.summary.FileWriter(summaries_dir + '/test')
sess.run(tf.global_variables_initializer())

for step in range(num_steps + 1):
	sess.run(t_index_set_op)
	if step % (num_steps / 1000) == 0:
		summary, r_err1 = sess.run([merged, t_err])
		train_writer.add_summary(summary, step)
		sess.run(train1)
		r_err2, r_acc = sess.run([t_err, t_acc])
		print('step:', step, ', pre err: ', r_err1, ', acc:', r_acc, 'post err:', r_err2)
	else:
		sess.run(train1)

print('done')