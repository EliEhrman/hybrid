"""
knnseq.py

same as knnmatch (explained immediately) except that the samples are either sequences (up or down) or random

forked from knnmatch.py

This module uses a knn search to classify data into either repeated two halves or not.
(see in fork source explanantion below)

However, after training a NN to do the job it tests a knn altennative.
The original NN has two layers. This module takes the output of the first layer, once trained
and uses that as the embedding of the entire sample data set. It then takes
a random batch, creates embeddings from the first layer and finds the k closest
based on this. The labels of those closes are averaged (reciprocal rank) to produce
a prediction.

Forked from match.py. Initial comment follows:
This module is about the simplest example I managed to create of a neural network in Tensorflow

Its main purpose is for use as a starting point for more ambitious experiments. The goal being that whenever
I ceate something I want to know that at inception it already can learn a simple function. Then as each stage is
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
import random
# import os
# import math
import collections


num_vals = 4
num_samples = 100000
batch_size = 1000
core_dim = 80
num_steps = 10000000 # 10000000
learn_rate = 40.0 # 4.0
acc_thresh = 0.9

# np.random.seed(1)

data = np.ndarray([num_samples, num_vals], dtype=np.float32)
truth = np.ndarray([num_samples, 2], dtype=np.float32)
for isamp in range(num_samples / 2):
	epsilon = random.uniform(0., 0.01)
	end = random.uniform(epsilon * 2.0, 1.0)
	start = random.uniform(0.0, end - epsilon)
	if random.uniform(0., 1.0) > 0.5:
		data[isamp * 2, :] = np.linspace(start, end, num_vals)
	else:
		data[isamp * 2, :] = np.linspace(end, start, num_vals)
	data[isamp*2+1, :]=np.random.uniform(start, end, num_vals)
	truth[isamp * 2] = [1.0, 0.0]
	truth[isamp * 2 + 1] = [0.0, 1.0]

last_train = num_samples * 9 / 10

t_data = tf.constant(data, name='data_pool')
t_truth = tf.constant(truth, name='truth_data_pool')
t_index = tf.Variable(tf.random_uniform([batch_size], 0, num_samples - 1, dtype=tf.int32), trainable=False, name='index_for_batching')

with tf.name_scope('nn_params'):
	wfc1 = tf.Variable(tf.random_normal([num_vals, core_dim], 0, 1e-3, dtype=tf.float32), name='t_wfc1')
	bfc1 = tf.Variable(tf.random_normal([core_dim], 0, 1e-3, dtype=tf.float32), name='t_bfc1')
	wfc2 = tf.Variable(tf.random_normal([core_dim, 2], 0, 1e-3, dtype=tf.float32), name='t_wfc2')
	bfc2 = tf.Variable(tf.random_normal([2], 0, 1e-3, dtype=tf.float32), name='t_bfc2')

with tf.name_scope('data_setup'):
	t_index_set_op = tf.assign(t_index, tf.random_uniform([batch_size], 0, last_train - 1, dtype=tf.int32))
	t_index_set_op_test = tf.assign(t_index, tf.random_uniform([batch_size], last_train, num_samples - 1, dtype=tf.int32))
	t_index_set_op_dummy = tf.assign(t_index, tf.range(batch_size, dtype=tf.int32))
	t_batch = tf.gather(t_data, t_index, name='gather_data')
	t_batch_truth = tf.gather(t_truth, t_index, name='gather_truth')

with tf.name_scope('nn'):
	fc1 = tf.sigmoid(tf.matmul(t_batch, wfc1, name='matmul_fc1') + bfc1, name='sig_fc1')
	pred = tf.sigmoid(tf.matmul(fc1, wfc2, name='matmul_fc2') + bfc2, name='sig_fc2')

with tf.name_scope('evaluation'):
	t_err = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=pred,
																   labels=t_batch_truth), name='x_entropy')
	train1 = tf.train.GradientDescentOptimizer(learning_rate=learn_rate, name='GDO').minimize(t_err, name='GDOmin')
	tf.summary.scalar('Error', t_err)

	t_correct = tf.equal(tf.argmax(pred, 1, name='am_pred'), tf.argmax(t_batch_truth, 1, name='am_truth'), name='eq_correct')
	t_acc = tf.reduce_mean(tf.cast(t_correct, tf.float32), name='acc_mean')
	tf.summary.scalar('Accuracy', t_acc)

num_ks = 111
with tf.name_scope('knn'):
	t_knn_db = tf.nn.l2_normalize(tf.sigmoid(tf.matmul(t_data, wfc1, name='matmul_knn_fc1') + bfc1, name='fc1_knn'), dim=1, name='l2_norm_knn_db')
	t_fc1_norm = tf.nn.l2_normalize(fc1, dim=1, name='l2_norm_knn_q_batch')
	AllCDs = tf.matmul(t_fc1_norm, t_knn_db, transpose_b=True, name='make_cds')
	t_bestCDs, t_bestCDIDxs = tf.nn.top_k(AllCDs, num_ks, sorted=True, name='top_k')
	t_knn_vals = tf.gather(t_knn_db, t_bestCDIDxs, name='gather_knn')
	t_rank_raw = tf.add(tf.range(num_ks, dtype=tf.float32), 2.0, name='rank_knn')
	t_rank_sum = tf.reduce_sum(tf.reciprocal(t_rank_raw))
	t_rank = tf.transpose(tf.tile(tf.expand_dims(t_rank_raw, 0), [2, 1]))
	t_knn_truth_by_idx = tf.gather(t_truth, t_bestCDIDxs, name='truth_by_idx')
	t_knn_truth = tf.divide(tf.reduce_sum(tf.divide(t_knn_truth_by_idx, t_rank), axis=1, name='mean_truth_by_idx'), t_rank_sum)
	t_corr_by_knn = tf.reduce_sum(tf.abs(tf.subtract(t_knn_truth, t_batch_truth)), name='corr_by_knn') / (batch_size * 2)
	t_knn_acc_raw = tf.equal(tf.argmax(t_knn_truth, 1, name='knn_am_pred'), tf.argmax(t_batch_truth, 1, name='knn_am_truth'), name='knn_eq_correct')
	t_knn_acc = tf.reduce_mean(tf.cast(t_knn_acc_raw, tf.float32), name='knn_acc_mean')

"""
The argument to Session in the following produces an output that shows to which device each and every variable
is allocated. On my machine with a cpu and gpu almost everything is allocated to gpu:0
Accuracy and Error are not. (perhaps because of the summary)
"""

# sess = tf.Session(config=tf.ConfigProto(
#     log_device_placement=True))
sess = tf.Session()
merged = tf.summary.merge_all()
summaries_dir = '/tmp'
train_writer = tf.summary.FileWriter(summaries_dir + '/train',
                                      sess.graph)
# test_writer = tf.summary.FileWriter(summaries_dir + '/test')
# tf.set_random_seed(1)
sess.run(tf.global_variables_initializer())

r_rank, r_rank_sum = sess.run([t_rank, t_rank_sum])


for step in range(num_steps + 1):
	sess.run(t_index_set_op)
	if step % (num_steps / 1000) == 0:
		summary, r_err1 = sess.run([merged, t_err])
		train_writer.add_summary(summary, step)
		sess.run(train1)
		r_err2, r_acc = sess.run([t_err, t_acc])
		sess.run(t_index_set_op_test)
		r_test_acc = sess.run(t_acc)
		print('step:', step, ', pre err: ', r_err1, ', acc:', r_acc, 'post err:', r_err2, 'test acc:', r_test_acc)
		if r_acc > acc_thresh:
			print('optimization done.')
			break
	else:
		sess.run(train1)

sess.run(t_index_set_op_test)
# r_bestCDIDxs, r_knn_truth, r_batch_truth, r_bestCDs, r_fc1, r_fc1_norm, r_knn_vals, \
# 		r_wfc1, r_bfc1, r_index, r_batch, r_knn_truth_by_idx \
# 		= sess.run([t_bestCDIDxs, t_knn_truth, t_batch_truth, t_bestCDs, fc1, t_fc1_norm, t_knn_vals,
# 					wfc1, bfc1, t_index, t_batch, t_knn_truth_by_idx])
print('knn error:', sess.run(t_corr_by_knn), 'knn accuracy:', sess.run(t_knn_acc))

sess.close()

print('all done')