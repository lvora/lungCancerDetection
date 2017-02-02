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

"""Preprocessing of DICOM Images for Kaggle Data Science 2017

"""
import numpy as np
import cv2
import os
import csv
import pprint
#import tensorflow as tf

im_dir = '/home/charlie/Downloads/stage1'
label_dir = '/home/charlie/Downloads/stage1_labels.csv'

class DicomDict:
    ''' DicomDict
    DicomDict constructs the dictionary of all DICOM images,
    their labels, collects useful metrics, and does basic error
    checking
    '''
    def __init__(self, path_to_images, path_to_labels):
        self.path_to_images = path_to_images
        self.path_to_labels = path_to_labels
        self.dicom_dict = self.__build_dict(self.path_to_images, 
                self.path_to_labels)

    def __build_dict(self, path_to_images, path_to_labels):
        dicom_dict = dict()
        for root, dirs, files in os.walk(path_to_images):
            path = root.split(os.sep)
            top_id = os.path.basename(root)
            dicom_dict[top_id] = {}
            dicom_dict[top_id]['slice_count'] = len(files)
            for file in files:
                slice_id, _ = os.path.splitext(file)
                dicom_dict[top_id][slice_id] = {}
        return self.__assign_labels(path_to_labels, dicom_dict)

    def __assign_labels(self, path_to_labels, dicom_dict):
        with open(path_to_labels) as cf:
            reader = csv.DictReader(cf,delimiter=',')
            for row in reader:
                dicom_dict['%s' % row['id']]['cancer'] = row['cancer']
        return dicom_dict

    def save(self, path_to_save): 
        ''' DicomDict.save(path_to_save = STR)
        This dumps the dictionary to a csv file.  Only useful for 
        troubleshooting
        '''
        w = csv.writer(open(os.path.join(path_to_save,'dicom_dict_dump_kds17.csv'),'w'))
        for key, val in self.dicom_dict.items():
            w.writerow([key, val])
        print('dicom_dict_dump_kds17.csv has been saved in %s' % path_to_save)

class DicomImage:
    def __init__(self, im_dir):
        self.im_id = im_id

class DicomSlice:
    def __init__(self, slice_id):
        self.slice_id = slice_id

def main(argv=None):

    x = DicomDict(im_dir, label_dir)
    x.save('/home/charlie/projects/kds17')

if __name__ == '__main__':
    main()
    
