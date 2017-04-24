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

import kds17_tf_model_lv as tf_model
import kds17_io as kio
import kds_Qstatespace as SA
import kds_MetaQNN as QNN

def main(argv = None):
    im_dir = r'C:\Users\jeremy\projects\stage1\100'
    label_dir = r'C:\Users\jeremy\projects\stage1\stage1_labels.csv'
    pickle_dir = r'C:\Users\jeremy\projects\kaggle_pickle1' 

    io = kio.DicomIO(pickle_dir, im_dir, label_dir) 
    if 0:
        tf_model.run_train(io, 10)
    else:
        sa = SA.getstate()
        NUMSTATES, S = sa.countstates()
        NUMACTIONS,A = sa.countactions()
        
        epsilon = ((500,1),(100,0.9),(100,0.8),(100,0.7),(100,0.6),(100,0.5),(100,0.4),
                   (100,0.3),(200,0.2),(300,0.1))
        replay_memory = []
        for i in range(len(epsilon)):
            print('epsilon',epsilon[i])
            Q,Sprime,Aprime,replay_memory = QNN.Q_learning(io,NUMSTATES,NUMACTIONS,epsilon[i][0],epsilon[i][1],S,A,replay_memory)
        #tf_model.run_test(io, 10)

if __name__ == '__main__':
    main()
