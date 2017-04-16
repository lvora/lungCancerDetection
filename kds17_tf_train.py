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

import kds17_tf_model as tf_model
import kds17_io as kio


def main(argv = None):
    im_dir = '/home/charlie/data/lung_cancer/stage1'
    label_dir = '/home/charlie/data/lung_cancer/stage1_labels.csv'
    pickle_dir = '/home/charlie/data/kaggle_pickle/'

    io = kio.DicomIO(pickle_dir, im_dir, label_dir) 
    tf_model.run_train(io, 100000)
    #tf_model.run_test(io, 10)

if __name__ == '__main__':
    main()
