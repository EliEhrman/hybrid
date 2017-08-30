"""
knnlearn.py


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
from tensorflow.python import debug as tf_debug


num_vals = 4
num_samples = 100000
batch_size = 1000
core_dim = 8
num_steps = 10000000
learn_rate = 4.0
knn_learn_rate = 0.0005
acc_thresh = 0.70
num_knn_steps = 100000
num_ks = 11
knn_bad_factor = 1.0e-8
knn_epsilon = 1.0e-8

np.random.seed(1)

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
	data[isamp*2+1, :]=data[isamp*2, :]
	data[isamp * 2 + 1, random.randint(0, num_vals-1)] = random.uniform(start, end)
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
	op_index_set = tf.assign(t_index, tf.random_uniform([batch_size], 0, last_train - 1, dtype=tf.int32), name='op_index_set')
	op_index_set_test = tf.assign(t_index, tf.random_uniform([batch_size], last_train, num_samples - 1, dtype=tf.int32), name='op_index_set_test')
	op_index_set_dummy = tf.assign(t_index, tf.range(batch_size, dtype=tf.int32))
	t_batch = tf.gather(t_data, t_index, name='t_batch')
	t_batch_truth = tf.gather(t_truth, t_index, name='t_batch_truth')

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

with tf.name_scope('knn'):
	full_knn_shape = [batch_size, num_ks, core_dim]
	t_knn_db_prenorm = tf.sigmoid(tf.matmul(t_data, wfc1, name='matmul_knn_fc1') + bfc1, name='t_knn_db_prenorm')
	t_knn_db = tf.nn.l2_normalize(t_knn_db_prenorm, dim=1, name='t_knn_db')
	t_fc1_norm = tf.nn.l2_normalize(fc1, dim=1, name='t_fc1_norm')
	AllCDs = tf.matmul(t_fc1_norm, t_knn_db, transpose_b=True, name='AllCDs')
	t_bestCDs, t_bestCDIDxs = tf.nn.top_k(AllCDs, num_ks, sorted=True, name='top_k')
	# The purpose of the next two lines is to create a variable that is a starting point for the graph such that backprop
	# learning does not go beyond the knn index and tries to learn the whole database
	# We create an untrainable variable with the shape of the indixes
	# v_knn_best_ids = tf.Variable(tf.zeros([batch_size, num_ks], dtype=tf.int32), trainable=False)
	v_knn_best_ids = tf.Variable(tf.random_uniform([batch_size, num_ks], 0, num_samples-1, dtype=tf.int32), trainable=False)
	# The op to do the assign will end up creatng the knn db of all training instances and doing the knn on it to get the top
	# k results. This is created for each of the batch_size elements. (The transform to create the db will only happen onc,
	# the CDs will be created in one big matrix operation and the sorting to find the knn will happen once for each
	# of the elements in batch_size
	# Shape = [batch_size, num_ks]
	op_knn_best_ids_set = tf.assign(v_knn_best_ids, t_bestCDIDxs)
	# create the vals, k for each element in batch, gathered from the original data. Shape=[batch_size, num_ks, num_vals]
	t_knn_data = tf.gather(t_data, v_knn_best_ids, name='t_knn_data')
	# since this is the wrong shape we must reshape as if we just have more data items.
	t_raw_vals_by_knn = tf.reshape(t_knn_data, [batch_size * num_ks, num_vals])
	## t_knn_vals_unnorm = tf.t_knn_db_prenorm(t_data, t_bestCDIDxs, name='gather_knn_best_from_raw_data')
	# For the sake of efficiency we do no take the data directly from t_data and then perform the matmul followed by
	# the gather op but rather gather from the raw data and then perform the matmul etc.
	## v_knn_vals = tf.Variable(tf.zeros([batch_size, num_ks, core_dim], dtype=tf.float32), trainable=False, name='v_knn_vals')
	## op_knn_vals_set = tf.assign(v_knn_vals, tf.gather(t_data, t_bestCDIDxs), name='op_knn_vals_set')
	# Create the input for the knn error and accuracy value by applying the original transform that created the knn, *minus the l2 norm*
	# and the reshape back so that the first dimension is batch_size. Shape=[batch_size, num_ks, core_dim]
	t_knn_vals_unnorm = tf.reshape(tf.sigmoid(tf.matmul(t_raw_vals_by_knn,
														wfc1, name='matmul_knn_unnorm_wfc1') + bfc1,
														name='knn_vals_unnorm'), [batch_size, num_ks, core_dim])
	# the basic rank tensor which requires broadcasting before use. Shape=[num_ks]
	t_rank_raw = tf.add(tf.range(num_ks, dtype=tf.float32), 2.0, name='rank_knn')
	t_rank_sum = tf.reduce_sum(tf.reciprocal(t_rank_raw), name='t_rank_sum')
	# Now produce a tensor we can use to weight the knn results. Shape = [num_ks, 2]. All we've done is take the [2, 3, 4 ...]
	# vector and turned it into [[2, 2], [3, 3], [4, 4] ...] so that the divide won't complain
	# it's a sort of explicit broadcast needed for shapes that the automatic broadcast can't handle
	# Shape = [num_ks, 2]
	t_rank = tf.tile(tf.expand_dims(t_rank_raw, 1), [1, 2], name='t_rank')
	# The truth value here is the original labels of the nearest neighbors that have been selected. Shape = [batch_size, num_ks, 2]
	t_knn_truth_by_idx = tf.gather(t_truth, v_knn_best_ids, name='t_knn_truth_by_idx')
	# weight the truth values by the rank of the result. i.e. closer neighbors. We are calculating an average truth value.
	# What we have calculated here is the truth value for each element of the batch as predicted by the knn. Shape=[batch_size, 2]
	t_knn_truth = tf.divide(tf.reduce_sum(tf.divide(t_knn_truth_by_idx, t_rank), axis=1, name='mean_truth_by_idx'), t_rank_sum)
	# calculate the error for each knn result at the output nodes/truth/label level. Shape=[batch_size, 2]
	# This is calculated AFTER we have already done a weighted average on the knn results to get a prediction per batch element
	t_knn_diff = tf.abs(tf.subtract(t_knn_truth, t_batch_truth), name='t_knn_diff')
	# Calculate the average error. Simple mean except that we calculate for each of the two output nodes. Shape=()
	t_corr_by_knn = tf.reduce_sum(t_knn_diff / (batch_size * 2), name='t_corr_by_knn')
	# calculate the raw accuracy by comparing the larger of the two nodes (weighted average) with the lale in batch truth. Shape=[batch_size]
	t_knn_acc_raw = tf.equal(tf.argmax(t_knn_truth, 1, name='knn_am_pred'), tf.argmax(t_batch_truth, 1, name='knn_am_truth'), name='t_knn_acc_raw')
	# cast from bool to float and calculate the mean. Shape = ()
	t_knn_acc = tf.reduce_mean(tf.cast(t_knn_acc_raw, tf.float32), name='knn_acc_mean')

	# t_rank_broadcast = tf.transpose(tf.tile(tf.expand_dims(t_rank_raw, 0), [core_dim, 1]), name='broadcast_for_avg')

	# create a version of t_batch_truth, the labels of the batch that is broadcast so that we can create a tensor of matches (marker/bool)
	# Shape of original was [batch_size, 2]. Now Shape=[batch_size, num_ks, 2]
	t_batch_truth_broadcast = tf.tile(tf.expand_dims(t_batch_truth, 1), [1, num_ks, 1], name='t_batch_truth_broadcast')
	# The top k action before produced k closests indices. Some have the correct expected label anod some do not. We create a tensor of booleans
	# which tells us if the selected index is the correct lablel. This has shape [batch_size, num_ks] since each element
	# in the batch will have a vector of correct and incorrect indices
	# The ground truth value is the same for all k within the batch element, so we need to 'broadcast' or repeat the value across a newly created dimension
	# we use expand_dims and tile ops to do this. We need to do this again for different reasons in the next few tensors
	# t_good_k = tf.equal(tf.argmax(t_knn_truth_by_idx, 2), tf.tile(tf.expand_dims(tf.argmax(t_batch_truth, 1), 1), [1, num_ks], name='tile_for_batch'), name='good_k')
	t_good_k = tf.equal(tf.argmax(t_knn_truth_by_idx, 2), tf.argmax(t_batch_truth_broadcast, 2), name='t_good_k')
	# we need a broadcast of the above so that we can multiply the delta below either toward or away based on the good of the knn result
	# Shape=[batch_size, num_ks, core_dim]
	t_good_k_broadcast =  tf.tile(tf.expand_dims(t_good_k, -1), [1, 1, core_dim], name='t_good_k_broadcast')
	# we create a rank vector for each element in the batch. This is 1 / r + 2 where r is the rank. This is repeated for each element of the batch. Shape = [batch_size, num_ks]
	t_batch_rank = tf.tile(tf.expand_dims(tf.reciprocal(t_rank_raw), 0), [batch_size, 1], name='t_batch_rank')
	# now broadcast to another dimension, so that we can divide each of the core_dim. Shape=[batch_size, num_ks, core_dim]
	t_batch_rank_broadcast = tf.tile(tf.expand_dims(t_batch_rank, -1), [1, 1, core_dim], name='t_batch_rank_broadcast')
	# since the above will be used as weights for a weighted average we need a sum along the num_ks dimentsion. Shape=[batch_size, core_dim]
	t_batch_rank_sum = tf.reduce_sum(t_batch_rank_broadcast, axis=1, name='t_batch_rank_sum')
	## we need to zero out the rank tensor where t_good_k was false so that it doesn't figure in the final averaging. Shape = [batch_size, num_ks]
	##t_good_rank = tf.where(t_good_k, t_batch_rank, tf.zeros([batch_size, num_ks]), name='t_good_rank')
	## In order to be able to use the above as a factor, we need to apply it to each of the core_dim dimensions,
	## so we repeat the values along the core_dim. Shape = [batch_size, num_ks, core_dim]
	##t_good_rank_broadcast = tf.tile(tf.expand_dims(t_good_rank, -1), [1, 1, core_dim], name='t_good_rank_broadcast')
	## We need to sum t_good_rank calculated above. However, we will need the value broadcast to each of core_dim values in a batch element.
	## Shape = [batch_size, core_dim]
	##t_good_rank_sum = tf.tile(tf.expand_dims( tf.reduce_sum(t_good_rank, axis=1, name='good_rank_sum'), -1), [1, core_dim])
	# braodcast the original batch that was input to the knn op so that we can get a diff on each dim. Shape=[batch_size, num_ks, core_dim]
	t_fc1_broadcast = tf.tile(tf.expand_dims(fc1, 1), [1, num_ks, 1], name='t_fc1_broadcast')
	# Create a tensor of deltas for each core_dim value. Shape=[batch_size, num_ks, core_dim]
	# We basically want to move towards good values and away from bad values.
	t_knn_diffs = tf.abs(tf.subtract(t_knn_vals_unnorm, t_fc1_broadcast), name='t_knn_diffs')
	t_knn_deltas = tf.where(t_good_k_broadcast, t_knn_diffs, knn_bad_factor * tf.reciprocal(t_knn_diffs + knn_epsilon), name='t_knn_deltas')
	# We are finally ready to get the average (weighted by rank) coordinates of the good (matching) k closest neighbors
	# N.B. We do not used the normed values to get the average but go back to the unnnormed values
	# Shape = [batch_size, core_dim]
	t_knn_deltas_avg = tf.divide(tf.reduce_sum(tf.multiply(t_knn_deltas, t_batch_rank_broadcast), axis=1, name='sum_knn_by_idx'), t_batch_rank_sum, name='t_knn_deltas_avg')
	# Not done yet. We have NaN's in the result. We want to identify them and replace those guys with the orginal value
	# The reason for this is that we will use the avg in the error function that drives the backprop
	# The code paradigm for getting t_knn_avg[:, 0] is as in the following line
	# Update. I have found (without Internet confirmation yet) that I can use python indexing syntax
	# t_avg_slice = tf.reshape(tf.slice(t_knn_avg, [0, 0], [batch_size, 1]), [batch_size])
	##t_avg_slice = t_knn_avg[:, 0]
	# Now copy in the original value where the value is NaN (due to divide by zero) and use the rest
	# Shape = [batch_size, core_dim]
	##t_knn_desired = tf.Variable(tf.zeros([batch_size, core_dim], dtype=tf.float32), trainable=False, name='t_knn_desired')
	##op_knn_desired = tf.assign(t_knn_desired, tf.where(tf.is_nan(t_avg_slice), fc1, t_knn_avg), name='op_knn_desired')
	t_knn_error1 = tf.abs(t_knn_deltas_avg, name='t_knn_error1')
	t_knn_error = tf.reduce_mean(t_knn_error1, name='t_knn_error')
	train_knn = tf.train.GradientDescentOptimizer(learning_rate=knn_learn_rate, name='GDO_knn').minimize(t_knn_error1, name='GDOmin_knn')

with tf.name_scope('sanity'):
	t_wfc1_sum = tf.reduce_sum(tf.abs(wfc1))
	t_bfc1_sum = tf.reduce_sum(tf.abs(bfc1))

"""
The argument to Session in the following produces an output that shows to which device each and every variable
is allocated. On my machine with a cpu and gpu almost everything is allocated to gpu:0
Accuracy and Error are not. (perhaps because of the summary)
"""

# sess = tf.Session(config=tf.ConfigProto(
#     log_device_placement=True))
sess = tf.Session()
# sess = tf_debug.LocalCLIDebugWrapperSession(sess, ui_type="curses")
merged = tf.summary.merge_all()
summaries_dir = '/tmp'
train_writer = tf.summary.FileWriter(summaries_dir + '/train',
                                      sess.graph)
# test_writer = tf.summary.FileWriter(summaries_dir + '/test')
tf.set_random_seed(1)
sess.run(tf.global_variables_initializer())

r_rank, r_rank_sum = sess.run([t_rank, t_rank_sum])
sess.run(op_index_set)
r_knn_truth_by_idx, r_good_k, r_batch_truth, r_batch_rank, r_knn_error, r_good_k_broadcast, r_batch_rank_broadcast, \
					r_batch_rank_sum, r_fc1_broadcast, r_knn_deltas, r_knn_deltas_avg, r_fc1, r_knn_vals_unnorm, r_knn_diffs\
		= sess.run([t_knn_truth_by_idx, t_good_k, t_batch_truth, t_batch_rank, t_knn_error, t_good_k_broadcast,
					t_batch_rank_broadcast, t_batch_rank_sum, t_fc1_broadcast, t_knn_deltas, t_knn_deltas_avg, fc1, t_knn_vals_unnorm, t_knn_diffs])


for step in range(num_steps + 1):
	sess.run(op_index_set)
	if step % (num_steps / 1000) == 0:
		summary, r_err1 = sess.run([merged, t_err])
		train_writer.add_summary(summary, step)
		sess.run(train1)
		r_err2, r_acc = sess.run([t_err, t_acc])
		sess.run(op_index_set_test)
		r_test_acc = sess.run(t_acc)
		print('step:', step, ', pre err: ', r_err1, ', acc:', r_acc, 'post err:', r_err2, 'test acc:', r_test_acc)
		if r_acc > acc_thresh:
			print('optimization done.')
			break
	else:
		sess.run(train1)

for step in range(num_knn_steps + 1):
	if step % (num_knn_steps / 1000) == 0:
		sess.run(op_index_set_test)
		sess.run(op_knn_best_ids_set)
		r_knn_error = sess.run(t_knn_error)
		r_knn_acc = sess.run(t_knn_acc)
		print('knn step:', step, ', err: ', r_knn_error, 'test acc:', r_knn_acc)
		print('sanity: wfc1 sum =', sess.run(t_wfc1_sum), 'bfc1 sum =', sess.run(t_bfc1_sum))

	sess.run(op_index_set)
	sess.run(op_knn_best_ids_set)
	sess.run(train_knn)

sess.run(op_index_set_test)
# r_bestCDIDxs, r_knn_truth, r_batch_truth, r_bestCDs, r_fc1, r_fc1_norm, r_knn_vals, \
# 		r_wfc1, r_bfc1, r_index, r_batch, r_knn_truth_by_idx \
# 		= sess.run([t_bestCDIDxs, t_knn_truth, t_batch_truth, t_bestCDs, fc1, t_fc1_norm, t_knn_vals,
# 					wfc1, bfc1, t_index, t_batch, t_knn_truth_by_idx])
print('knn error:', sess.run(t_corr_by_knn), 'knn accuracy:', sess.run(t_knn_acc))

sess.close()

print('all done')