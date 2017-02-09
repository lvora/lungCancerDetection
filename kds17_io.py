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

"""Input/Output handling of DICOM Images for Kaggle Data Science 2017

"""
import sys
import os
import time
from tqdm import tqdm
import threading as th
import pickle


class DicomIO:
    ''' DicomIO
    DicomIO handles loading saving and checking of the dicom pickle files

    Args:
        pickle_dir: valid path to directory containing pickle files or
                    where they should be saved
    Retruns:
        DicomIO.<args>
        DicomIO.path_exist: Boolean indicating whether the path exists
        DicomIO.list: List containing the names of all files in pickle_dir

    '''
    def __init__(self, pickle_dir):
        self.pickle_dir = pickle_dir
        self.path_exist = self.__check_path()
        self.list = []

    def __check_path(self):
        if os.path.isdir(self.pickle_dir):
            self.list = [f for f in os.listdir(self.pickle_dir) if os.path.isfile(os.path.join(self.pickle_dir, f))] 
            return True
        else:
            os.mkdir(self.pickle_dir)
            self.__check_path()

    def save(self, dicomBatch): 
        print('Saving %s in %s' % (dicomBatch.name, self.pickle_dir))
        with open(os.path.join(self.pickle_dir,dicomBatch.name+'.pkl',), 'wb')as f:
            pickle.dump(dicomBatch,f)
        self.__check_path()

        
    def load(self, pickle_name=None): 

        if pickle_name is not None:
            print('Loading %s from %s' % (pickle_name, self.pickle_dir))
            with open(os.path.join(self.pickle_dir,pickle_name), 'rb')as f:
                return pickle.load(f)
        else:
            self.__check_path()
            batch_list = []
            for k in self.list:
                print('Loading everything from %s' % (self.pickle_dir))
                with open(os.path.join(self.pickle_dir,k), 'rb')as f:
                    batch_list.append(pickle.load(f))
            return batch_list                    


