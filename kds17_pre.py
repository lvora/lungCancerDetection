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
import dicom
import os
import csv
import pprint
from tqdm import tqdm
from multiprocessing import Pool
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
        self.dicom_dict = self.__build_dict()

    def __build_dict(self):
        dicom_dict = dict()
        try:
            for root, dirs, files in os.walk(self.path_to_images):
                path = root.split(os.sep)
                top_id = os.path.basename(root)
                if len(files) > 0:
                    dicom_dict[top_id] = {}
                    dicom_dict[top_id]['slice_count'] = len(files)
        except:
            print('Dictionary Failed')
        return self.__assign_labels(dicom_dict)

    def __assign_labels(self,dicom_dict):
        try:
            with open(self.path_to_labels) as cf:
                reader = csv.DictReader(cf,delimiter=',')
                for row in reader:
                    dicom_dict['%s' % row['id']]['cancer'] = row['cancer']
        except:
            print('Labels not assigned')
        return self.__filtered_dict(dicom_dict)

    def __filtered_dict(self, dicom_dict):
        del_count = 0
        for x in list(dicom_dict.keys()):
            for y in list(dicom_dict[x].keys()):
                if x in dicom_dict.keys() and 'cancer' not in dicom_dict[x].keys():
                    del_count += 1
                    del dicom_dict[x]
        print('Dictionary Built with %i items deleted for missing labels or images' % del_count)
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
    def __init__(self, path_to_image,label, im_id):
        self.im_id = im_id
        self.label = label
        self.path_to_image = path_to_image
        self.slices = self.__load_image()

    def __load_image(self):
        slices = [dicom.read_file(self.path_to_image + '/' + s) for s in os.listdir(self.path_to_image)]
        slices.sort(key = lambda x: int(x.ImagePositionPatient[2]))
        try:
            slice_thickness = np.abs(slices[0].ImagePositionPatient[2]-slices[1].ImagePositionPatient[2])
        except:
            slice_thickness = np.abs(slices[0].SliceLocation - slices[1].SliceLocation)
        for s in slices:
            s.SliceThickness = slice_thickness
        return slices

def helper(job_args):
    return DicomImage(*job_args)

def main(argv=None):

    x = DicomDict(im_dir, label_dir)
    #pprint.pprint(x.dicom_dict)

    path_list = [os.path.join(x.path_to_images,k) for k, v in x.dicom_dict.items()]
    print(path_list)
    label_list = [list(j for i, j in v.items() if i == 'cancer')[0] for k, v in x.dicom_dict.items()]
    id_list = list(x.dicom_dict.keys())
    
    job_args = [(path_list, label_list, id_list)]

    p = Pool()
    image_list = p.map(helper, job_args)

if __name__ == '__main__':
   main() 
