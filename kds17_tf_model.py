# Copyright 2017 GATECH ECE6254 KDS17 TEAM. All Rights Reserved.
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

import tensorflow as tf
from six.moves import xrange
import time
import numpy as np
import kds17_tf_input as tf_input
import kds17_io as kio

IMAGE_SIZE = tf_input.IMAGE_SIZE
NUM_CLASSES = 2
BATCH_SIZE = 1
MAX_STEPS = 1000000
MOVING_AVERAGE_DECAY = 0.9999     
NUM_EPOCHS_PER_DECAY = 350.0      
NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN = 1000
LEARNING_RATE_DECAY_FACTOR = 1e-07  
INITIAL_LEARNING_RATE = 5e-07       
FILTER_SIZE = 8 
IN_CHANNEL = 1
OUT_CHANNEL = 1
DTYPE = tf.float32

def __activation_summary(x):
   tensor_name = x.op.name
   tf.summary.histogram(tensor_name + '/activations', x)
   tf.summary.scalar(tensor_name + '/sparsity', tf.nn.zero_fraction(x))

def __var_on_cpu_mem(name, shape, initializer=None, dtype=tf.float32):
    with tf.device('/cpu:0'):
        return tf.get_variable(name,shape,initializer=initializer,dtype=dtype)

def inference(images):
    with tf.variable_scope('conv1') as scope:
        kernel = __var_on_cpu_mem('weights', 
                        [FILTER_SIZE,FILTER_SIZE,FILTER_SIZE,IN_CHANNEL, OUT_CHANNEL], 
                        initializer=tf.truncated_normal_initializer(
                            stddev=5e-2,
                            dtype=DTYPE))
        conv = tf.nn.conv3d(images, kernel, [1,1,1,1,1], padding='SAME')
        biases = __var_on_cpu_mem('biases',
                        [IN_CHANNEL*OUT_CHANNEL],
                        initializer=tf.constant_initializer(0.0))
        pre_activation = tf.nn.bias_add(conv,biases)
        conv1 = tf.nn.relu(pre_activation, name=scope.name)
        __activation_summary(conv1)

    pool1 = tf.nn.max_pool3d(conv1, ksize=[1,2,2,2,1], strides=[1,2,2,2,1],
                            padding='SAME', name='pool1')

    in_channel_squeeze = tf.squeeze(pool1, [4])

    norm1 = tf.nn.lrn(in_channel_squeeze, BATCH_SIZE, bias=1.0, alpha=0.001 / 16.0, 
                        beta=0.75, name='norm1')

    with tf.variable_scope('local2') as scope:
        reshape = tf.reshape(pool1, [BATCH_SIZE, -1])
        dim = reshape.get_shape()[1].value
        weights = __var_on_cpu_mem('weights', [dim, 16])
        biases = __var_on_cpu_mem('biases', 
                        [16], 
                        initializer=tf.constant_initializer(0.1))
        local2 = tf.nn.relu(tf.matmul(reshape, weights) + biases, name=scope.name)
        __activation_summary(local2)

    with tf.variable_scope('local3') as scope:
        weights = __var_on_cpu_mem('weights', [16, 8])
        biases = __var_on_cpu_mem('biases', 
                        [8], 
                        initializer=tf.constant_initializer(0.1), 
                        dtype=DTYPE)
        local3 = tf.nn.relu(tf.matmul(local2, weights) + biases, name=scope.name)
        __activation_summary(local3)

    with tf.variable_scope('softmax_linear') as scope:
        weights = __var_on_cpu_mem('weights', [8, NUM_CLASSES])
        biases = __var_on_cpu_mem('biases', 
                        [NUM_CLASSES], 
                        initializer=tf.constant_initializer(0.0))
        softmax_linear = tf.add(tf.matmul(local3, weights), biases, name=scope.name)
        __activation_summary(softmax_linear)

    return softmax_linear

def loss(logits, labels):
    labels = tf.cast(labels, tf.int32)
    cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(
            logits, labels, name='cross_entropy_per_example')
    cross_entropy_mean = tf.reduce_mean(cross_entropy, name='cross_entropy')
    tf.add_to_collection('losses', cross_entropy_mean)
    return tf.add_n(tf.get_collection('losses'), name='total_loss')

def __add_loss_summaries(total_loss):
    loss_averages = tf.train.ExponentialMovingAverage(0.9, name='avg')
    losses = tf.get_collection('losses')
    loss_averages_op = loss_averages.apply(losses + [total_loss])
    for l in losses + [total_loss]:
        tf.summary.scalar(l.op.name +' (raw)', l)
        tf.summary.scalar(l.op.name, loss_averages.average(l))
    return loss_averages_op

def train(total_loss, global_step):
    num_batches_per_epoch = NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN / BATCH_SIZE 
    decay_steps = int(num_batches_per_epoch * NUM_EPOCHS_PER_DECAY)

    lr = tf.train.exponential_decay(INITIAL_LEARNING_RATE,
                                  global_step,
                                  decay_steps,
                                  LEARNING_RATE_DECAY_FACTOR,
                                  staircase=True)
    tf.summary.scalar('learning_rate', lr)

    loss_averages_op = __add_loss_summaries(total_loss)

    with tf.control_dependencies([loss_averages_op]):
        opt = tf.train.GradientDescentOptimizer(lr)
        grads = opt.compute_gradients(total_loss)

    apply_gradient_op = opt.apply_gradients(grads, global_step=global_step)

    for var in tf.trainable_variables():
        tf.summary.histogram(var.op.name, var)

    for grad, var in grads:
        if grad is not None:
            tf.summary.histogram(var.op.name + '/gradients', grad)

    variable_averages = tf.train.ExponentialMovingAverage(
                MOVING_AVERAGE_DECAY, global_step)
    variables_averages_op = variable_averages.apply(tf.trainable_variables())

    with tf.control_dependencies([apply_gradient_op, variables_averages_op]):
        train_op = tf.no_op(name='train')

    return train_op

def run_train(DicomIO, max_steps = 10):
    with tf.Graph().as_default():
        global_step = tf.Variable(0, trainable=False)
        feeder = tf_input.DicomFeeder(DicomIO)
        images, labels = feeder.next_batch(BATCH_SIZE)
        print(labels)
        logits = inference(images)
        losss = loss(logits, labels)
        train_op = train(losss, global_step)
        summary_op = tf.summary.merge_all()
        init = tf.global_variables_initializer()
        session_config = tf.ConfigProto(log_device_placement=False, allow_soft_placement=True)
        session_config.gpu_options.allocator_type = 'BFC'
        sess = tf.Session(config=session_config)
        sess.run(init)
        tf.train.start_queue_runners(sess=sess)
        start_step = 0
        for step in xrange(start_step, start_step+max_steps):
            start_time = time.time()
            _, loss_value = sess.run([train_op, losss])
            duration = time.time() - start_time
            print('Duration: %.3f Loss: %.3f' % (duration,loss_value))
            assert not np.isnan(loss_value), 'Model diverged with loss = NaN'

def __partition(im):
    im_q = []
    x0, x1 = tf.split(1, 2, im)
    for x in [x0, x1]:
        y0, y1 = tf.split(2, 2, x)
        for y in [y0, y1]:
            z0, z1 = tf.split(3, 2, y)
            im_q.append(z0)
            im_q.append(z1)
    return im_q

def run_test(DicomIO, max_steps = 10):
    with tf.Graph().as_default():
        global_step = tf.Variable(0, trainable=False)
        feeder = tf_input.DicomFeeder(DicomIO)
        images, labels = feeder.next_batch(BATCH_SIZE)
        init = tf.global_variables_initializer()
        session_config = tf.ConfigProto(log_device_placement=False, allow_soft_placement=True)
        session_config.gpu_options.per_process_gpu_memory_fraction = 0.70
        sess = tf.Session(config=session_config)
        sess.run(init)
        partition = sess.run([part])
        
        
        for im in partition:
            print(im.get_shape())

