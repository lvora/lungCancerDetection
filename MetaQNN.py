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


import kds_Qstatespace as SA

import tensorflow as tf
import numpy as np
import random

from six.moves import xrange
import time
import numpy as np
import kds17_tf_input as tf_input
import os
import kds17_io as kio

IMAGE_SIZE = tf_input.IMAGE_SIZE
NUM_CLASSES = 2
BATCH_SIZE = 1
#MAX_STEPS = 1000000
MAX_STEPS = 1000
MOVING_AVERAGE_DECAY = 0.99
NUM_EPOCHS_PER_DECAY = 350.0
NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN = 1000
LEARNING_RATE_DECAY_FACTOR = 1e-07
INITIAL_LEARNING_RATE = 5e-04
DECAY = 0.4
FILTER_SIZE = 8
FILTER_SIZE_2 = 5
FILTER_SIZE_3 = 3
IN_CHANNEL = 1
OUT_CHANNEL = 1
IN_CHANNEL_2 = 1
OUT_CHANNEL_2 = 1
IN_CHANNEL_3 = 1
OUT_CHANNEL_3 = 1
DROPOUT_VAL = 0.99
DTYPE = tf.float32

alpha = 0.01
K_ER = 10

folder = 1
train_dir_root = r'C:\Users\jeremy\projects\kds_train1'

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

def __add_loss_summaries(total_loss):
    loss_averages = tf.train.ExponentialMovingAverage(0.9, name='avg')
    losses = tf.get_collection('losses')
    loss_averages_op = loss_averages.apply(losses + [total_loss])
    for l in losses + [total_loss]:
        tf.summary.scalar(l.op.name + '_raw', l)
        tf.summary.scalar(l.op.name, loss_averages.average(l))
    return loss_averages_op

def Q_learning(DicomIO,NUMSTATES,NUMACTIONS,M,epsilon,statespace,A):
    replay_memory = []
    Q = tf.Variable(tf.constant(0.5, shape=[NUMSTATES, NUMSTATES]),name='Q')
    for episode in range(1,M):
        S,U = SAMPLE_NEW_NETWORK(epsilon,Q,statespace,A)
        print('Network Layout: ',[statespace[i] for i in S])
        accuracy = run_train(DicomIO, S, statespace,U)
        replay_memory.append((S,U,accuracy))
        for memory in range(K_ER):
            S_sample, U_sample, accuracy_sample = replay_memory[int(random.uniform(0,len(replay_memory)))]
            Q = UPDATE_Q_VALUES(Q,S_sample,U_sample,accuracy_sample)

    return Q,S,U

def SAMPLE_NEW_NETWORK(epsilon, Q,statespace,A):
    S = [statespace[0][1]]
    U = [0]
    while U[-1] != statespace[-1][1]:
        alpha = random.uniform(0,1)
        if alpha > epsilon:

            u = tf.arg_max(Q[0],dimension=1)
            sprime = TRANSITION(S[-1],u)
        else:
            u = A[S[-1]][int(random.uniform(0,len(A[S[-1]])-1))]
            sprime = TRANSITION(S[-1],u)
        U.append(u)
        S.append(sprime)
            
    return S,U

def UPDATE_Q_VALUES(Q,S,U,accuracy):
    global alpha
    
    Q[S[-1],U[-1]].assign((1-alpha)*Q[S[-1],U[-1]] + alpha*accuracy)
    for i in reversed(range(len(S)-2)):
        Q[S[i],U[i]].assign((1-alpha)*Q[S[i],U[i]] + alpha*Q[S[i+1],U[i]])
    return Q

def TRANSITION(s,u):
    sprime = u
    return sprime


def MetaQNN(images, keep_prob,S,s):
    print(len(S))
    if len(S)>0:
        out1 = modelstep(images, S[0],s,keep_prob)
        if len(S)==1: return out1
    if len(S)>1:
        out2 = modelstep(out1, S[1],s,keep_prob)
        if len(S)==2: return out2
    if len(S)>2:
        out3 = modelstep(out2, S[2],s,keep_prob)
        if len(S)==3: return out3
    if len(S)>3:
        out4 = modelstep(out3, S[3],s,keep_prob)
        if len(S)==4: return out4
    if len(S)>4:
        out5 = modelstep(out4, S[4],s,keep_prob)
        if len(S)==5: return out5
    if len(S)>5:
        out6 = modelstep(out5, S[5],s,keep_prob)
        if len(S)==6: return out6            
    if len(S)>6:
        out7 = modelstep(out6, S[6],s,keep_prob)
        if len(S)==7: return out7
    if len(S)>7:
        out8 = modelstep(out7, S[7],s,keep_prob) 
        if len(S)==8: return out8
    if len(S)>8:
        out9 = modelstep(out8, S[8],s,keep_prob) 
        if len(S)==9: return out9
    if len(S)>9:
        out10 = modelstep(out9, S[9],s,keep_prob) 
        if len(S)==10: return out10
    if len(S)>10:
        out11 = modelstep(out10, S[10],s,keep_prob) 
        if len(S)==11: return out11
    if len(S)>11:
        out12 = modelstep(out11, S[11],s,keep_prob) 
        if len(S)==12: return out12
    if len(S)>12:
        out13 = modelstep(out12, S[12],s,keep_prob) 
        if len(S)==13: return out13


def modelstep(ins, S,s,keep_prob):
    if s[S][0] == 'C':
        FILTER_SIZE = s[S][2]
        l = s[S][3]
        IN_CHANNEL = 1
        OUT_CHANNEL = 1
        if 1:
            with tf.variable_scope(str(S)) as scope:
                kernel = __var_on_cpu_mem('weights',
                                  [FILTER_SIZE,
                                   FILTER_SIZE,
                                   FILTER_SIZE,
                                   IN_CHANNEL,
                                   OUT_CHANNEL],
                                  None,
                                  initializer=tf.truncated_normal_initializer(
                                      stddev=5e-2,
                                      dtype=DTYPE))
                
                conv = tf.nn.conv3d(ins,
                                    kernel,
                                    [l, l, l, l, l],
                                    padding='SAME')
                biases = __var_on_cpu_mem('biases',
                                          [IN_CHANNEL*OUT_CHANNEL],
                                          None,
                                          initializer=tf.constant_initializer(0.0))
                
                pre_activation = tf.nn.bias_add(conv, biases)
                out = tf.nn.relu(pre_activation, name=scope.name)
                __activation_summary(out)
                
                return(out)
        else:        
            with tf.variable_scope(str(S)) as scope:
                kernel = __var_on_cpu_mem('weights',
                                  [FILTER_SIZE,
                                   FILTER_SIZE,
                                   FILTER_SIZE,
                                   IN_CHANNEL,
                                   OUT_CHANNEL],
                                  None,
                                  initializer=tf.truncated_normal_initializer(
                                      stddev=5e-2,
                                      dtype=DTYPE))
                
                conv = tf.nn.conv3d(ins,
                                    kernel,
                                    [l, l, l, l, l],
                                    padding='SAME')
                biases = __var_on_cpu_mem('biases',
                                          [IN_CHANNEL*OUT_CHANNEL],
                                          None,
                                          initializer=tf.constant_initializer(0.0))
                
                pre_activation = tf.nn.bias_add(conv, biases)
                out = tf.nn.relu(pre_activation, name=scope.name)
                __activation_summary(out)
                
                return(out)
        
        
    if s[S][0] == 'P':
        k = s[S][2]
        stride = s[S][3]
        out = tf.nn.max_pool3d(ins,
                             ksize=[1, k, k, k, 1],
                             strides=[1, stride, stride, stride, 1],
                             padding='SAME',
                             name=str(S))
        return(out)
    
    if s[S][0] == 'DO':
        with tf.variable_scope(str(S)) as scope:
            drop_out = tf.nn.dropout(ins, keep_prob)
            __activation_summary(drop_out)
            return drop_out

    if s[S][0] == 'relu':
        with tf.variable_scope(str(S)) as scope:
            out = tf.nn.relu(ins, name=scope.name)
            __activation_summary(out)
            return out
        
    if s[S][0] == 'Terminate':
        
        x = ins.get_shape()[1]
        if 1:#x <= 8:
            with tf.variable_scope('softmax_linear') as scope:
                reshape = tf.reshape(ins, [BATCH_SIZE, -1])
                dim = reshape.get_shape()[1]
                weights = __var_on_cpu_mem('weights', [dim, NUM_CLASSES], DECAY)
                biases = __var_on_cpu_mem('biases',
                                          [NUM_CLASSES],
                                          None,
                                          initializer=tf.constant_initializer(0.0))
                softmax_linear = tf.add(tf.matmul(reshape, weights),biases,name=scope.name)
                __activation_summary(softmax_linear)
                return softmax_linear
        else:
            with tf.variable_scope('local2') as scope:
                reshape = tf.reshape(ins, [BATCH_SIZE, -1])
#                dim = (IMAGE_SIZE**3)/(8*8*8)
                dim = reshape.get_shape()[1]
                weights = __var_on_cpu_mem('weights', [dim, 256], DECAY)
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
                softmax_linear = tf.add(tf.matmul(local3, weights),biases,name=scope.name)
                __activation_summary(softmax_linear)
                # print('there')
                return softmax_linear
            
def loss(logits, labels):

    labels = tf.cast(labels, tf.int32)
    # print(logits.get_shape())
    # print(labels.get_shape())
    cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(
            logits=logits, labels=labels, name='cross_entropy_per_example')
    cross_entropy_mean = tf.reduce_mean(cross_entropy, name='cross_entropy')
    tf.add_to_collection('losses', cross_entropy_mean)
    return tf.add_n(tf.get_collection('losses'), name='total_loss')

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

def run_train(DicomIO,S,statespace,U, max_steps=10, logits_op=None):

    folder= 'S'+''.join([str(i)+'_' for i in S]) +'U'+ ''.join([str(i)+'_' for i in U])
    train_dir = train_dir_root+'\\'+str(folder)
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
        logits = MetaQNN(images, keep_prob, S,statespace) 
        #inference(images,keep_prob)
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
            _, loss_value, summary_str = sess.run([train_op,loss_val,summary_op],feed_dict=feed_dict)
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

            if step%10 == 0: #step%1000 == 0:
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
    return accuracy_eval

def main():
    debugS = [('C',1,8,1,1,8),
              ('P',2,2,2,8),
              ('C',3,5,1,1,4),
              ('P',4,2,2,4),
              ('C',5,3,1,1,4),
              ('P',6,2,2,1),
              ('Terminate',)
              ]
    sa = SA.getstate()
    NUMSTATES, S = sa.countstates()
    NUMACTIONS,A = sa.countactions()
    s = []
    for item in debugS:
        print(item)
        for item2 in S:        
            if item == item2[:-1]:
                s.append(item2[-1])
                print(S[item2[-1]])

    im_dir = r'C:\Users\jeremy\projects\stage1\100'
    label_dir = r'C:\Users\jeremy\projects\stage1\stage1_labels.csv'
    pickle_dir = r'C:\Users\jeremy\projects\kaggle_pickle1' 
    DicomIO = kio.DicomIO(pickle_dir, im_dir, label_dir)
    
    run_train(DicomIO, s, S)              

# if __name__ == '__main__':
     #main()
