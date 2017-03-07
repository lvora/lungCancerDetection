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


def main(argv=None):
    im_dir = '/home/charlie/kaggle_stage1' 
    label_dir = '/home/charlie/kaggle_stage1/stage1_labels.csv'
    pickle_dir = '/home/charlie/kaggle_pickles/' 
    io = kio.DicomIO(pickle_dir, im_dir, label_dir) 

    # io.build_batch()
        
    z = io.load_batch('batch_0.pkl')
    im = z.batch[0].image
    x = np.linspace(np.min, np.max, 1)
    print(x)
    
    print('max:%.2e min:%.2e' % (np.max(im), np.min(im))) 
    
    vis.animate(vis.rotate(im, 'anterposterior'))


if __name__ == '__main__':
    main()
