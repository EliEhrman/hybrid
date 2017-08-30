from __future__ import print_function
# import csv
import numpy as np
import tensorflow as tf
# import random
# import os
# import math
import collections

core_dim = 10
batch_size = 8
num_steps = 10000
num_samples = 100000

s_channel = collections.namedtuple('channel', 'num b_input data')


def create_net(in_channel_list, out_channel_list, params):
	b_incoming = True
	channel_agg = tf.zeros([batch_size, core_dim], dtype=tf.float32)
	for key in in_channel_list:
		channel = in_channel_list[key]
		data = channel.data
		ii = tf.sigmoid(tf.matmul(data, params[key + '_iiw']) + params[key + '_iib'])
		gate_dict = {}
		for gate_name in 'ifco':
			gate = tf.zeros([batch_size, core_dim], dtype=tf.float32)
			for key2 in in_channel_list:
				channel2 = in_channel_list[key2]
				data2 = channel2.data
				gate += tf.matmul(data2, params[key + '_' + gate_name + '_' + key2])
			tf.nn.bias_add(gate, params[key + '_' + gate_name])
			gate_dict[gate_name] = gate
		i = tf.sigmoid(gate_dict['i'], name=key + "_i_gate")
		f = tf.sigmoid(gate_dict['f'], name=key + "_f_gate")
		o = tf.sigmoid(gate_dict['o'], name=key + "_o_gate")
		c = tf.tanh(gate_dict['c'], name=key + "_c_gate")
		c = ii
		# c = f * ii + i * c
		channel_agg = tf.add(channel_agg, tf.matmul(c, params[key + '_aggw']))
	channel_agg = tf.add(channel_agg, params['aggb'])
	# fc_last = tf.nn.tanh(tf.matmul(channel_agg, params['wfc1']) + params['bfc1'])
	fc_last = channel_agg
	for key in out_channel_list:
		channel = out_channel_list[key]
		channel_out = tf.nn.tanh(tf.matmul(fc_last, params[key + '_selw']) + params[key + '_selb'])
		out_channel_list[key] = channel._replace(
			# data=tf.nn.relu(tf.matmul(o * channel_out, params[key + '_finw']) + params[key + '_finb']))
			data=tf.nn.relu(tf.matmul(channel_out, params[key + '_finw']) + params[key + '_finb']))

def create_weights(params, key, wshape):
	# params[key] = tf.get_variable(key, shape=wshape, initializer = tf.contrib.layers.xavier_initializer())
	params[key] = tf.get_variable(key, shape=wshape, initializer = tf.random_uniform_initializer(0, 2.0e-1))

def create_params(in_channel_list, out_channel_list):
	stddev = 1.0e-2
	params = {}
	for key in in_channel_list:
		channel = in_channel_list[key]
		channel_dim = channel.num
		wshape = [channel_dim, core_dim]
		create_weights(params, key + '_iiw', wshape)
		create_weights(params, key + '_iib', [core_dim])
		for gate_name in 'ifco':
			for key2 in in_channel_list:
				channel2 = in_channel_list[key]
				create_weights(params, key +  '_' + gate_name + '_' + key2, wshape)
			create_weights(params, key + '_' + gate_name, [core_dim])
		create_weights(params, key + '_aggw', [core_dim, core_dim])
	create_weights(params, 'aggb', [core_dim])
	create_weights(params, 'wfc1', [core_dim, core_dim])
	create_weights(params, 'bfc1', [core_dim])
	for key in out_channel_list:
		channel = out_channel_list[key]
		channel_dim = channel.num
		wshape = [core_dim, channel_dim]
		create_weights(params, key + '_selw', [core_dim, core_dim])
		create_weights(params, key + '_selb', [core_dim])
		create_weights(params, key + '_finw', wshape)
		create_weights(params, key + '_finb', [channel_dim])
	return params


in_channel_list = dict()
channel_dim = 6
in_channel_list['input'] = s_channel(
	num=channel_dim,
	b_input=True,
	data=tf.Variable(tf.random_uniform([batch_size, channel_dim], 0, 1.0, dtype=tf.float32), trainable=True)
)
# in_channel_list['ireg'] = s_channel(
# 	num=channel_dim,
# 	b_input=False,
# 	# data=tf.Variable(tf.random_uniform([batch_size, channel_dim], 0, 1.0, dtype=tf.float32), trainable=False)
# 	data=tf.Variable(tf.zeros([batch_size, channel_dim], dtype=tf.float32), trainable=False)
# )

out_channel_list = dict()
channel_dim = 2
out_channel_list['output'] = s_channel(
	num=channel_dim,
	b_input=True,
	data=tf.Variable(tf.random_uniform([batch_size, channel_dim], 0, 1.0, dtype=tf.float32))
)
# channel_dim = 6
# out_channel_list['oreg'] = s_channel(
# 	num=channel_dim,
# 	b_input=False,
# 	data=tf.Variable(tf.random_uniform([batch_size, channel_dim], 0, 1.0, dtype=tf.float32), trainable=False)
# )

params = create_params(in_channel_list, out_channel_list)
create_net(in_channel_list, out_channel_list, params)

num_vals = in_channel_list['input'].num
data = np.ndarray([num_samples, num_vals], dtype=np.float32)
truth = np.ndarray([num_samples, 2], dtype=np.float32)
for isamp in range(num_samples / 2):
	data[isamp*2, :num_vals/2]=np.random.rand(num_vals/2)
	data[isamp*2, num_vals/2:]=data[isamp*2, :num_vals/2]
	data[isamp*2+1, :]=np.random.rand(num_vals)
	# data[isamp * 2, :] = np.zeros(num_vals)
	# data[isamp * 2 + 1, :] = np.ones(num_vals)
	truth[isamp * 2] = [1.0, 0.0]
	truth[isamp * 2 + 1] = [0.0, 1.0]

t_data = tf.constant(data)
t_truth = tf.constant(truth)
t_index = tf.Variable(tf.random_uniform([batch_size], 0, num_samples - 1, dtype=tf.int32), trainable=False)
# t_index_set_op = tf.assign(t_index, tf.random_uniform([batch_size], 0, num_samples - 1, dtype=tf.int32))
t_index_set_op = tf.assign(t_index, tf.random_uniform([batch_size], 0, batch_size - 1, dtype=tf.int32))
t_batch = tf.gather(t_data, t_index)
t_batch_truth = tf.gather(t_truth, t_index)
t_set_input_op = tf.assign(in_channel_list['input'].data, t_batch)
ph_input = tf.placeholder(tf.float32, shape=[in_channel_list['input'].num])
# ph_ireg = tf.placeholder(tf.float32, shape=[in_channel_list['ireg'].num])

t_err = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=out_channel_list['output'].data,
															   labels=t_batch_truth))
train1 = tf.train.GradientDescentOptimizer(learning_rate=0.5).minimize(t_err)
opt = tf.train.GradientDescentOptimizer(learning_rate=0.05)
grads_and_vars = opt.compute_gradients(t_err, [
												params['input_iiw'], params['input_iib'],
												params['input_aggw'],
												params['aggb'],
												params['output_selw'], params['output_selb'],
												params['output_finw'], params['output_finb'],
												])
opt_op = opt.apply_gradients(grads_and_vars)

t_correct = tf.equal(tf.argmax(out_channel_list['output'].data, 1), tf.argmax(t_batch_truth, 1))
t_acc = tf.reduce_mean(tf.cast(t_correct, tf.float32))

sess = tf.Session()
sess.run(tf.global_variables_initializer())

for step in range(num_steps + 1):
	sess.run(t_index_set_op)
	sess.run(t_set_input_op)
	if step % (num_steps / 1000) == 0:
		r_index, r_batch, r_batch_truth, r_predict = sess.run(
			[t_index, t_batch, t_batch_truth, out_channel_list['output'].data])
		r_acc, r_err, r_correct = sess.run([t_acc, t_err, t_correct])
		r_grads_and_vars = sess.run([grads_and_vars])
		sess.run(opt_op)
		r_err2 = sess.run(t_err)
		print('step:', step, ', pre err: ', r_err, ', acc:', r_acc, 'post err:', r_err2)
	else:
		sess.run(opt_op)
		# sess.run(train1)

sess.close

print('done')
