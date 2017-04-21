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
import numpy as np
from six.moves import xrange

# Global Variables
IMAGE_SIZE = 80
NUM_OF_RAND_CROP_PER_IMAGE = 30

class DicomFeeder(object):
    def __init__(self, DicomIO):
        self.__batch_index = 0
        self.__epoch = 0
        self.__batch_file_list = DicomIO.batch_file_list
        self.__train_set, self.__eval_set = self.__generate_train_and_eval()
        self.__io = DicomIO

    def __generate_train_and_eval(self):
        np.random.shuffle(self.__batch_file_list)
        train_idx = int(len(self.__batch_file_list)//1.25)
        train_set = self.__batch_file_list[:train_idx] 
        eval_set = self.__batch_file_list[train_idx:]
        return train_set, eval_set 

    def next_batch(self, batch_size, from_eval_set):
        if from_eval_set:
            this_set = self.__eval_set
        else:
            this_set = self.__train_set

        set_len = len(this_set)

        if self.__batch_index > set_len-1:
            np.random.shuffle(this_set)
            self.__batch_index = 0
            self.__epoch += 1

        image_batch = self.__io.load_batch(
                this_set[self.__batch_index], squelch=True).batch

        images = [np.expand_dims(x.image, axis=-1) for x in image_batch]
        labels = [int(x.label) for x in image_batch]

        combined = list(zip(images, labels))
        np.random.shuffle(combined)
        images[:], labels[:] = zip(*combined)
        
        self.__batch_index += 1

        if from_eval_set:
            images_agg = []
            labels_agg = []
            rand_num = np.random.randint(1, 3)
            #return rand_crop(images[:batch_size]), labels[:batch_size]  #ORIGINAL
            for i in xrange(batch_size):
                images_agg.append(images[rand_num])
                for j in range(NUM_OF_RAND_CROP_PER_IMAGE):
                    labels_agg.append(labels[rand_num])
            return rand_crop(images_agg), labels_agg
        else:

            labels_augmented = []
            for lab in labels[:batch_size]:
                for j in range(NUM_OF_RAND_CROP_PER_IMAGE):
                    labels_augmented.append(lab)

            return rand_crop(images[:batch_size]), labels_augmented


def rand_crop(im):
    im_slice = []
    for i in im:
        for j in range(NUM_OF_RAND_CROP_PER_IMAGE):
            shape = np.array(i.shape)[:3]
            offset = (np.random.rand(1, 3)-0.5)[0]
            start = shape-IMAGE_SIZE
            start = np.array(start//2+start*offset//2, dtype=np.int)
            end = start+IMAGE_SIZE
            im_slice.append(i[start[0]:end[0],
                              start[1]:end[1],
                              start[2]:end[2],
                              :])
    return im_slice


def placeholder_inputs(batch_size):
    image_placeholder = tf.placeholder(tf.float32, shape=([batch_size, 
                                                           IMAGE_SIZE, 
                                                           IMAGE_SIZE, 
                                                           IMAGE_SIZE, 
                                                           1])) 
    label_placeholder = tf.placeholder(tf.int32, shape=(batch_size, ))
    keep_prob = tf.placeholder(tf.float32)
    return image_placeholder, label_placeholder, keep_prob


def fill_feed_dict(feeder, image_pl, label_pl, batch_size, for_eval, dropout, do_val):
    image_feed, label_feed = feeder.next_batch(batch_size, for_eval)

    if for_eval:
        do  = 1
    else:
        do = do_val

    feed_dict = {
            image_pl: image_feed,
            label_pl: label_feed,
            dropout: do
            }

    return feed_dict
