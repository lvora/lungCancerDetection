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

# Global Variables
IMAGE_SIZE = 256
num_threads = 4


class DicomFeeder(object):
    def __init__(self, DicomIO):
        self.__batch_index = 0
        self.__image_index = 0
        self.__epoch = 0
        self.__batch_file_list = DicomIO.batch_file_list
        self.__train_set, self.__eval_set = self.__generate_train_and_eval()
        self.__io = DicomIO
        self.__images, self.__labels, self.__set_len = self.__next_batch()
        self.__eval = False

    def __generate_train_and_eval(self):
        np.random.shuffle(self.__batch_file_list)
        train_idx = int(len(self.__batch_file_list)//1.25)
        train_set = self.__batch_file_list[:train_idx] 
        eval_set = self.__batch_file_list[train_idx:]
        return train_set, eval_set 

    def next_batch(self, batch_size, from_eval_set=False):
        image, label = self.__next_image(from_eval_set=from_eval_set)
        image_batch, label_batch = tf.train.shuffle_batch(
                                        [image, label], 
                                        batch_size=batch_size, 
                                        num_threads=num_threads, 
                                        capacity=4+3*batch_size, 
                                        min_after_dequeue=4)
        return image_batch, tf.reshape(label_batch, [batch_size])

    def __next_image(self, from_eval_set=False):
        if from_eval_set and not self.__eval:
            self.__images, self.__labels, self.__set_len = self.__next_batch(
                    from_eval_set=from_eval_set)
            self.__eval = from_eval_set

        if self.__image_index > len(self.__images)-1:
            self.__images, self.__labels, self.__set_len = self.__next_batch(
                    from_eval_set=from_eval_set)
            self.__image_index = 0

        image = tf.image.per_image_standardization(
                    self.__images[self.__image_index])
        image = tf.expand_dims(image, -1)
        label = self.__labels[self.__image_index]

        self.__image_index += 1
        if from_eval_set:
            return self.__mid_crop(image), label
        else:
            return self.__rand_transpose(self.__rand_crop(image)), label

    def __next_batch(self, from_eval_set=False):
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
                this_set[self.__batch_index]).batch

        images = [tf.cast(x.image, dtype=tf.float32) for x in image_batch]
        labels = [tf.cast(int(x.label), dtype=tf.int32) for x in image_batch]

        combined = list(zip(images, labels))
        np.random.shuffle(combined)
        images[:], labels[:] = zip(*combined)
        
        self.__batch_index += 1

        return images, labels, set_len

    def __rand_crop(self, im):
        im_slice = tf.random_crop(im, [IMAGE_SIZE, IMAGE_SIZE, IMAGE_SIZE, 1])
        return im_slice

    def __mid_crop(self, im):
        start = (tf.shape(im)-IMAGE_SIZE)//2
        im_slice = tf.slice(im, [start[0], start[1], start[2]],
                            [1, IMAGE_SIZE, IMAGE_SIZE, IMAGE_SIZE])
        return im_slice

    def __rand_transpose(self, im):
        first_3 = np.random.permutation(3)
        perm = np.append(first_3, 3)
        im_trans = tf.transpose(im, perm=perm)
        return im_trans


def placeholder_inputs(batch_size):
    image_placeholder = tf.placeholder(tf.float32, 
                                       shape=([batch_size, None, None, None])) 
    label_placeholder = tf.placeholder(tf.int32, shape=(batch_size, ))
    return image_placeholder, label_placeholder


def fill_feed_dict(feeder, image_pl, label_pl, batch_size, for_eval=False):
    image_feed, label_feed = feeder.next_batch(batch_size, 
                                               from_eval_set=for_eval)
    
    feed_dict = {
            image_pl: image_feed,
            label_pl: label_feed,
            }

    return feed_dict 

