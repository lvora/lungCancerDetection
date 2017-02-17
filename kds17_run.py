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

from kds17_pre import *
from kds17_io import *

import kds17_pre as pre
import kds17_io as kio
import numpy as np



def main(argv = None):
    im_dir = '/home/charlie/kaggle_data/stage1'
    label_dir = '/home/charlie/kaggle_data/stage1_labels.csv'
    pickle_dir = '/home/charlie/kaggle_pickle/'

    x = pre.DicomDict(im_dir, label_dir)
    io = kio.DicomIO(pickle_dir)
    for i,args in enumerate(tqdm(x.batched_job_args)):
        y = pre.DicomBatch(args, 'batch_%i'% i)
        threads = []
        for im in y.batch:
            t = pre.ProcessDicomBatch(DicomImage=im)
            t.setDaemon(True)
            threads.append(t)
        [t.start() for t in threads]
        print('Preprocessing images for batch_%i'% i)
        [t.join() for t in threads]
        io.save(y)

    print(z[0].batch[0].image.shape)


    

if __name__ == '__main__':
    main()
