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

import kds17_io as kio
import tensorflow as tf
import numpy as np

# Global Variables
IMAGE_SIZE = 256
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

    def next_image(self, from_eval_set=False):
        if from_eval_set and not self.__eval:
            self.__images, self.__labels, self.__set_len = self.__next_batch(
                    from_eval_set=from_eval_set)
            self.__eval = from_eval_set

        if self.__image_index > len(self.__images)-1:
            self.__images, self.__labels, self.__set_len = self.__next_batch(
                    from_eval_set=from_eval_set)
            self.__image_index = 0

        image = self.__images[self.__image_index]
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

        images = [tf.cast(x.image,dtype = tf.int32) for x in image_batch]
        labels = [tf.cast(x.label,dtype = tf.int32) for x in image_batch]

        combined = list(zip(images, labels))
        np.random.shuffle(combined)
        images[:], labels[:] = zip(*combined)
        
        self.__batch_index += 1

        return images, labels, set_len

    def __rand_crop(self, im):
        im_slice = tf.random_crop(im, [IMAGE_SIZE, IMAGE_SIZE, IMAGE_SIZE])
        return im_slice

    def __mid_crop(self, im):
        start = (tf.shape(im)-IMAGE_SIZE)//2
        end = start+IMAGE_SIZE
        im_slice = tf.slice(im, [start[0], start[1], start[0]],[IMAGE_SIZE, IMAGE_SIZE, IMAGE_SIZE])
        return im_slice

    def __rand_transpose(self, im):
        perm = np.random.permutation(3)
        im_trans = tf.transpose(im, perm=perm)
        return im_trans

def placeholder_inputs():
    image_placeholder = tf.placeholder(tf.float32, shape=([None, None, None])) 
    label_placeholder = tf.placeholder(tf.int32, shape=(1))
    return image_placeholder, label_placeholder

def fill_feed_dict(feeder, image_pl, label_pl, for_eval=False):
    image_feed, label_feed = feeder.next_image(from_eval_set=for_eval)
    
    feed_dict = {
            image_pl: image_feed,
            label_pl: label_feed,
            }

    return feed_dict 

def main(argv = None):
    im_dir = '/home/charlie/kaggle_data' 
    label_dir = '/home/charlie/kaggle_data/stage1_labels.csv'
    pickle_dir = '/home/charlie/kaggle_pickle/' 

    io = kio.DicomIO(pickle_dir, im_dir, label_dir) 
    feeder = DicomFeeder(io)
    image_pl, label_pl = placeholder_inputs()

    for i in range(10):
        x = fill_feed_dict(feeder, image_pl, label_pl)
        print(x[image_pl])

    for i in range(10):
        x = fill_feed_dict(feeder, image_pl, label_pl, for_eval=True)
        print(x[image_pl])

if __name__ == '__main__':
    main()
