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
import numpy as np
from matplotlib import pyplot as plt


def main(argv = None):
# This is the top level folder containing all scan folders
    im_dir = '/home/charlie/kaggle_stage1' 

# This is the path to the labels in csv
    label_dir = '/home/charlie/kaggle_stage1/stage1_labels.csv'

# This will be created if it does not exist
    pickle_dir = '/home/charlie/kaggle_pickles/' 

# Creating an IO object with working directories 
# im_dir and label_dir are not required after
# batches exist in your pickle_dir
    io = kio.DicomIO(pickle_dir, im_dir, label_dir) 

# This will save a dictionary.pkl in your pickle_dir
#    io.save_dict()

# This is demonstrating accessing an attribute from the DicomDict object
#    print(io.DicomDict.batch_size_limit)

# This will queue all valid image/label pairs in batches that are <= batch_size_limit
# then it will conduct all preprocessing in threads that should not exceed
# system memory. PSUTILs is used here and may fail.  Sorry if it does because I 
# am assuming it works.
#    io.build_batch()
        
# This is an example of loading all batches from your pickle_dir into a list of
# DicomBatch objects
    z = io.load_batch('batch_0.pkl')
        
# This is an example of loading an image from the batch attribute within the loaded
# DicomBatch object
    im = z.batch[0].image
    plt.hist(im.ravel(), 2000)
    plt.gca().set_yscale('log')
    plt.show()
        
# Let's look at it!
#    vis.animate(vis.rotate(im, 'anterposterior'))

if __name__ == '__main__':
    main()
