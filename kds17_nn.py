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
import kds17_vis as vis
import tensorflow as tf
import numpy as np

class DicomFeeder(object):
    def __init__(self, DicomIO):
        self.__batch_index = 0
        self.__image_index = 0
        self.__epoch = 0
        self.__batch_file_list = DicomIO.batch_file_list
        self.__train_set, self.__eval_set = self.__generate_train_and_eval()
        self.__io = DicomIO
        self.__images, self.__labels, self.__set_len = self.__next_batch()

    def __generate_train_and_eval(self):
        np.random.shuffle(self.__batch_file_list)
        train_idx = int(len(self.__batch_file_list)//1.25)
        train_set = self.__batch_file_list[:train_idx] 
        eval_set = self.__batch_file_list[train_idx:]
        return train_set, eval_set 

    def next_image(self, from_eval_set=False):
        '''Do something
        '''
        if from_eval_set:
            self.__images, self.__labels, self.__set_len = self.__next_batch(
                    from_eval_set=from_eval_set)

        if self.__image_index > len(self.__images)-1:
            self.__images, self.__labels, self.__set_len = self.__next_batch(
                    from_eval_set=from_eval_set)
            self.__image_index = 0

        image = self.__images[self.__image_index]
        label = self.__labels[self.__image_index]

        self.__image_index += 1
        return image, label



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

        images = [tf.cast(x.image,dtype = tf.int16) for x in image_batch]
        labels = [tf.cast(x.label,dtype = tf.int16) for x in image_batch]

        combined = list(zip(images, labels))
        np.random.shuffle(combined)
        images[:], labels[:] = zip(*combined)
        
        self.__batch_index += 1

        return images, labels, set_len
    
def placeholder_inputs():
    '''do something
    '''
    image_placeholder = tf.placeholder(tf.float32, shape=([None, None, None])) 
    label_placeholder = tf.placeholder(tf.int32, shape=(1))
    return image_placeholder, label_placeholder

def fill_feed_dict(feeder, image_pl, label_pl, for_eval=False):
    '''do something
    '''
    image_feed, label_feed = feeder.next_image(from_eval_set=for_eval)
    
    feed_dict = {
            image_pl: image_feed,
            label_pl: label_feed,
            }

    return feed_dict 

def main(argv = None):
    im_dir = '/home/charlie/kaggle_sample' 
    label_dir = '/home/charlie/kaggle_sample/stage1_labels.csv'
    pickle_dir = '/home/charlie/kaggle_pickles/' 
    io = kio.DicomIO(pickle_dir, im_dir, label_dir) 
    feeder = DicomFeeder(io)
    image_pl, label_pl = placeholder_inputs()
    for i in range(1000):
        x = fill_feed_dict(feeder, image_pl, label_pl)


if __name__ == '__main__':
    main()
