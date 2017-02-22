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

import kds17_tf_input as tf_input
import tensorflow as tf
import numpy as np

def train():
    with tf.Graph().as_default():
        image_pl, label_pl = placeholder_inputs()



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
