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
import pandas as pd
import random
import os

def main(argv = None):
    im_dir = r'C:\Users\jeremy\projects\stage1'
    label_dir = r'C:\Users\jeremy\projects\stage1\stage1_labels.csv'
    pickle_dir = r'C:\Users\jeremy\projects\kaggle_pickle' 
    csv_file = r'C:\Users\jeremy\projects\MetaQNN_accuracy.csv'
    train_dir_root = r'C:\Users\jeremy\projects\kds_train1'
    Q_csv_file = r'C:\Users\jeremy\projects\kds_train1\Q.csv'


    io = kio.DicomIO(pickle_dir, im_dir, label_dir) 
    if 0:
        tf_model.run_train(io, 10)
    else:
        sa = SA.getstate()
        NUMSTATES, S = sa.countstates()
        NUMACTIONS,A = sa.countactions()
        
           
        epsilon = ((1,500,1),(10,500,0.8),(10,500,0.6),(10,500,0.4),(15,500,0.3),(15,500,0.2),(15,500,0.1),(1,1500,0.1))
        # accuracy= []
        # accuracy_track = [[i,random.uniform(0,1)] for i in range(10)]
        accuracy_track_it = 0
        for i in range(len(epsilon)):
            print('epsilon',epsilon[i])
            
            Q,Sprime,Aprime,accuracy_track,episode = QNN.Q_learning(io,NUMSTATES,NUMACTIONS,epsilon[i][0],epsilon[i][2],S,A,MAX_STEPS=epsilon[i][1])#epsilon[i][1])
        #tf_model.run_test(io, 10)
            
            accuracy_track_it
            df_acc = pd.DataFrame(accuracy_track,columns=['iteration','accuracy'])
            df_acc['iteration'] = accuracy_track_it+df_acc['iteration']
            accuracy_track_it=df_acc['iteration'].iloc[-1]+1
            df_acc['epsilon'] = pd.Series(epsilon[i][2], index =df_acc.index)
            
            if os.path.exists(csv_file):
                df = pd.read_csv(csv_file, header = 0)
                df = pd.concat([df,df_acc])
                df = df.drop_duplicates(subset=['epsilon','iteration'],keep='last')
                df = df.reset_index()
                df.drop(['index'],axis=1,inplace=True)
            else:
                df = df_acc
            
            df.to_csv(csv_file,index=False, header=1)
                         
            if os.path.exists(train_dir_root+'\stop.txt'):
                return Q,Sprime,Aprime,accuracy_track,episode
                
    return Q,Sprime,Aprime,accuracy_track,episode
    
if __name__ == '__main__':
    Q,Sprime,Aprime,accuracy_track,episode = main()
    
    print('Q: ',Q)
    print('Last S: ',Sprime)
    print('Last A: ',Aprime)
    print('Accuracy_track: ',accuracy_track)
    print('Episode Finished: ', episode)
