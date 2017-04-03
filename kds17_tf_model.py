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
import os
import kds17_io as kio

IMAGE_SIZE = tf_input.IMAGE_SIZE
NUM_CLASSES = 2
BATCH_SIZE = 2
#MAX_STEPS = 1000000
MAX_STEPS = 10
MOVING_AVERAGE_DECAY = 0.99
NUM_EPOCHS_PER_DECAY = 350.0
NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN = 1000
LEARNING_RATE_DECAY_FACTOR = 1e-07
INITIAL_LEARNING_RATE = 5e-04
DECAY = 0.4
FILTER_SIZE_1 = 1
FILTER_SIZE_2 = 3
FILTER_SIZE_3 = 5
IN_CHANNEL = 1
OUT_CHANNEL = 1
IN_CHANNEL_2 = 1
OUT_CHANNEL_2 = 1
IN_CHANNEL_3 = 1
OUT_CHANNEL_3 = 1
DROPOUT_VAL = 0.99
DTYPE = tf.float32
train_dir = '/Users/lipivora/Documents/train_dir'


def __activation_summary(x):
    tensor_name = x.op.name
    tf.summary.histogram(tensor_name + '/activations', x)
    tf.summary.scalar(tensor_name + '/sparsity', tf.nn.zero_fraction(x))


def __var_on_cpu_mem(name, shape, decay, initializer=None, dtype=tf.float32):
    with tf.device('/cpu:0'):
        x = tf.get_variable(name,
                            shape,
                            initializer=initializer,
                            dtype=dtype)
    if decay is not None:
        reg = tf.multiply(tf.nn.l2_loss(x), decay, name='l2_regularization')
        tf.add_to_collection('losses', reg)
    return x


def inference(images,keep_prob):
    # Each layer of convolution will have multiple filters
    # Kernel 1 is a 1 by 1 convolution
    kernel1 = __var_on_cpu_mem('weights11',
                               [FILTER_SIZE_1,
                                FILTER_SIZE_1,
                                FILTER_SIZE_1,
                                IN_CHANNEL,
                                OUT_CHANNEL],
                               None,
                               initializer=tf.truncated_normal_initializer(
                                   stddev=5e-2,
                                   dtype=DTYPE))
    # Kernel 2 is a 3 by 3 convolution
    kernel2 = __var_on_cpu_mem('weights12',
                               [FILTER_SIZE_2,
                                FILTER_SIZE_2,
                                FILTER_SIZE_2,
                                IN_CHANNEL,
                                OUT_CHANNEL],
                               None,
                               initializer=tf.truncated_normal_initializer(
                                   stddev=5e-2,
                                   dtype=DTYPE))

    # Kernel 3 is a 5 by 5 convolution
    kernel3 = __var_on_cpu_mem('weights13',
                               [FILTER_SIZE_3,
                                FILTER_SIZE_3,
                                FILTER_SIZE_3,
                                IN_CHANNEL,
                                OUT_CHANNEL],
                               None,
                               initializer=tf.truncated_normal_initializer(
                                   stddev=5e-2,
                                   dtype=DTYPE))

    with tf.variable_scope('conv1') as scope:

        conv11 = tf.nn.conv3d(images,
                            kernel1,
                            [1, 1, 1, 1, 1],
                            padding='SAME')

        conv12= tf.nn.conv3d(images,
                            kernel2,
                            [1, 1, 1, 1, 1],
                            padding='SAME')

        conv13 = tf.nn.conv3d(images,
                            kernel3,
                            [1, 1, 1, 1, 1],
                            padding='SAME')

        # We need to concatenate the output of the various convolutions:
        conv = tf.concat(0, [conv11, conv12])
        conv = tf.concat(0, [conv, conv13])
        biases = __var_on_cpu_mem('biases',
                                  [IN_CHANNEL*OUT_CHANNEL],
                                  None,
                                  initializer=tf.constant_initializer(0.0))
        pre_activation = tf.nn.bias_add(conv, biases)
        conv1 = tf.nn.relu(pre_activation, name=scope.name)
        __activation_summary(conv1)


    pool1 = tf.nn.max_pool3d(conv1,
                             ksize=[1, 2, 2, 2, 1],
                             strides=[1, 2, 2, 2, 1],
                             padding='SAME',
                             name='pool1')
    new_dim = BATCH_SIZE * 3  # 3 is number of filters in any layer

    with tf.variable_scope('conv2') as scope:


        conv21 = tf.nn.conv3d(pool1,
                            kernel1,
                            [1, 1, 1, 1, 1],
                            padding='SAME')

        conv22= tf.nn.conv3d(pool1,
                            kernel2,
                            [1, 1, 1, 1, 1],
                            padding='SAME')

        conv23 = tf.nn.conv3d(pool1,
                            kernel3,
                            [1, 1, 1, 1, 1],
                            padding='SAME')
        conv2 = tf.concat(0, [conv21, conv22])
        conv2 = tf.concat(0, [conv2, conv23])

        biases = __var_on_cpu_mem('biases',
                                  [IN_CHANNEL * OUT_CHANNEL],
                                  None,
                                  initializer=tf.constant_initializer(0.0))

        pre_activation = tf.nn.bias_add(conv2, biases)
        conv2 = tf.nn.relu(pre_activation, name=scope.name)
        __activation_summary(conv2)

    pool2 = tf.nn.max_pool3d(conv2,
                             ksize=[1, 2, 2, 2, 1],
                             strides=[1, 2, 2, 2, 1],
                             padding='SAME',
                             name='pool1')
    new_dim *= 3


    with tf.variable_scope('local2') as scope:
        reshape = tf.reshape(pool2, [new_dim, -1])
        dim = (IMAGE_SIZE**3)/(8*8)
        weights = __var_on_cpu_mem('weights3', [dim, 256], DECAY)
        biases = __var_on_cpu_mem('biases',
                                  [256],
                                  None,
                                  initializer=tf.constant_initializer(0.1))
        drop_out = tf.nn.dropout(reshape, keep_prob)
        local2 = tf.nn.relu(tf.matmul(drop_out, weights) + biases,
                            name=scope.name)
        __activation_summary(local2)

    with tf.variable_scope('local3') as scope:
        weights = __var_on_cpu_mem('weights', [256, 8], DECAY)
        biases = __var_on_cpu_mem('biases',
                                  [8],
                                  None,
                                  initializer=tf.constant_initializer(0.1),
                                  dtype=DTYPE)

        local3 = tf.nn.relu(tf.matmul(local2, weights) + biases,
                            name=scope.name)
        __activation_summary(local3)

    with tf.variable_scope('softmax_linear') as scope:
        weights = __var_on_cpu_mem('weights', [8, NUM_CLASSES], DECAY)
        biases = __var_on_cpu_mem('biases',
                                  [NUM_CLASSES],
                                  None,
                                  initializer=tf.constant_initializer(0.0))
        softmax_linear = tf.add(tf.matmul(local3, weights),
                                biases,
                                name=scope.name)
        __activation_summary(softmax_linear)

    return softmax_linear


def loss(logits, labels):
    labels = tf.cast(labels, tf.int32)
    cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(
            logits=logits, labels=labels, name='cross_entropy_per_example')
    cross_entropy_mean = tf.reduce_mean(cross_entropy, name='cross_entropy')
    tf.add_to_collection('losses', cross_entropy_mean)
    return tf.add_n(tf.get_collection('losses'), name='total_loss')


def __add_loss_summaries(total_loss):
    loss_averages = tf.train.ExponentialMovingAverage(0.9, name='avg')
    losses = tf.get_collection('losses')
    loss_averages_op = loss_averages.apply(losses + [total_loss])
    for l in losses + [total_loss]:
        tf.summary.scalar(l.op.name + '_raw', l)
        tf.summary.scalar(l.op.name, loss_averages.average(l))
    return loss_averages_op

def evaluate(logits,labels):
    labels_agg = []
    logits_agg = tf.reduce_sum(logits,0,True)
    labels_agg.append(labels[0])
    correct = tf.nn.in_top_k(logits_agg,labels_agg, 1)
    #num_pred = tf.reduce_sum(tf.cast(correct, tf.int32))
    print('In evaluate')
    return correct

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
        opt = tf.train.AdamOptimizer(lr)
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


def run_train(DicomIO, max_steps=10, logits_op=None):
    feeder = tf_input.DicomFeeder(DicomIO)
    with tf.Graph().as_default():

        if tf.gfile.Exists(train_dir):
            print('Checkpoint')
            ckpt = tf.train.get_checkpoint_state(train_dir)
            global_step = tf.Variable(int(ckpt.model_checkpoint_path.split('/')[-1].split('-')[-1]), trainable=False)
        else:
            global_step = tf.Variable(0, trainable=False)

        images, labels, keep_prob = tf_input.placeholder_inputs(BATCH_SIZE)
#        tf.summary.image('images', images)
        logits = inference(images,keep_prob)
        loss_val = loss(logits, labels)
        train_op = train(loss_val, global_step)
        eval_op = evaluate(logits,labels)
        saver = tf.train.Saver(tf.global_variables())
        summary_op = tf.summary.merge_all()
        init = tf.global_variables_initializer()
        session_config = tf.ConfigProto(log_device_placement=False,
                                        allow_soft_placement=True)
        session_config.gpu_options.allocator_type = 'BFC'
        sess = tf.Session(config=session_config)
        sess.run(init)

        if tf.gfile.Exists(train_dir):
            tf.train.Saver().restore(sess, ckpt.model_checkpoint_path)
            start_step = int(ckpt.model_checkpoint_path.split('/')[-1].split('-')[-1])
        else:
            tf.gfile.MakeDirs(train_dir)
            start_step = 0

        tf.train.start_queue_runners(sess=sess)
        summary_writer = tf.summary.FileWriter(train_dir, sess.graph)
        for step in xrange(start_step, start_step+max_steps):
            start_time = time.time()
            feed_dict = tf_input.fill_feed_dict(feeder,
                                                images,
                                                labels,
                                                BATCH_SIZE,
                                                False,
                                                keep_prob,
                                                DROPOUT_VAL)
            _, loss_value, summary_str = sess.run([train_op,
                                                   loss_val,
                                                   summary_op],
                                                  feed_dict=feed_dict)
            duration = time.time() - start_time
            assert not np.isnan(loss_value), 'Model diverged with loss = NaN'
            examples_per_sec = BATCH_SIZE / duration
            sec_per_batch = float(duration)
            format_str = ('step %d, loss = %.2f (%.1f examples/sec; %.3f '
                          'sec/batch)')
            print(format_str % (step,
                                loss_value,
                                examples_per_sec,
                                sec_per_batch))

            if step % 10 == 0:
                summary_writer.add_summary(summary_str, step)
                checkpoint_path = os.path.join(train_dir, 'model.ckpt')
                saver.save(sess, checkpoint_path, global_step=step)

            if step%1000 == 0:
                num_correct = 0
                for i in xrange(10):  # 10 here can be replaced with number of evaluation images or portion of it
                    feed_dict = tf_input.fill_feed_dict(feeder,
                                                        images,
                                                        labels,
                                                        BATCH_SIZE,
                                                        True,
                                                        keep_prob,DROPOUT_VAL)
                    pred, loss_value = sess.run([eval_op, loss_val], feed_dict=feed_dict)
                    num_correct = num_correct + pred[0]
                accuracy_eval = num_correct / 10
                format_str = ('EVALUATION SET: step %d, loss = %.2f, accuracy=%.2f')
                print(format_str % (step,
                                    loss_value,
                                    accuracy_eval))
