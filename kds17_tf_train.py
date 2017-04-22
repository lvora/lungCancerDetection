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

#import kds17_tf_model as tf_model
import kds17_io as kio
import kds_Qstatespace as SA
import kds_MetaQNN as QNN

def main(argv = None):
    im_dir = '/home/charlie/kaggle_data/stage1'
    label_dir = '/home/charlie/kaggle_data/stage1_labels.csv'
    pickle_dir = '/home/charlie/kaggle_pickles/'

    io = kio.DicomIO(pickle_dir, im_dir, label_dir) 
    #tf_model.run_train(io, 10000)
    
    sa = SA.getstate()
    NUMSTATES, S = sa.countstates()
    NUMACTIONS,A = sa.countactions()
    
    epsilon = ((10,1),(1,0.9),(1,0.8),(1,0.7),(1,0.6),(1,0.5),(1,0.4),
               (1,0.3),(1,0.2),(1,0.1))
    for i in range(len(epsilon)):
        Q,Sprime,Aprime = QNN.Q_learning(io,NUMSTATES,NUMACTIONS,epsilon[i][0],epsilon[i][1],S,A)
    #tf_model.run_test(io, 10)

if __name__ == '__main__':
    main()
