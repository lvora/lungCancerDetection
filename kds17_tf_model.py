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
import kds17_tf_input as tf_input

def __activation_summary(x):
   tensor_name = x.op.name
   tf.summary.histogram(tensor_name + '/activations', x)
   tf.summary.scalar(tensor_name + '/sparsity', tf.nn.zero_fraction(x))

def inference(images):
    with tf.variable_scope('conv1') as scope:
        kernel = tf.get_variable('weights', 
                        shape=[64,64,3,64], 
                        initializer=tf.truncated_normal_initializer(
                            stddev=5e-2,
                            dtype=tf.float32))
        conv = tf.nn.conv2d(images, kernel, [1,1,1,1], padding='SAME')
        biases = tf.get_variiable('biases',
                        shape=[64],
                        initializer=tf.constant_initializer(0.0))
        pre_activation = tf.nn.bias_add(conv,bias)
        conv1 = tf.nn.relu(pre_activation, name=scope.name)
        __activation_summary(conv1)

    norm1 = tf.nn.lrn(conv1, 8, bias=1.0, alpha=0.001 / 64.0, 
                        beta=0.75, name='norm1')

    pool1 = tf.nn.max_pool(norm1, ksize=[1,8,8,1], strides=[1,4,4,1],
                            padding='SAME', name='pool1')
    print(pool1)



def main(argv = None):
    im_dir = '/home/charlie/kaggle_data'
    label_dir = '/home/charlie/kaggle_data/stage1_labels.csv'
    pickle_dir = '/home/charlie/kaggle_pickle/'
    with tf.Graph().as_default():
        images_pl, label_pl = tf_input.placeholder_inputs()
        logits = inference(images_pl)
        init = tf.glocal_variable_initializer()
        sess = tf.Session()
        sess.run(init)

        io = kio.DicomIO(pickle_dir, im_dir, label_dir)
        feeder = tf_input.DicomFeeder(io)
        feed_dict = tf_input.fill_feed_dict(feeder, image_pl, label_pl)
        _, loss_value = sess.run(feed_dict=feed_dict)

if __name__ == '__main__':
    main()
