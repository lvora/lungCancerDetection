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
from __future__ import division
import sys
import psutil
import dicom
import csv
import os
import scipy.ndimage
import time
from tqdm import tqdm
import numpy as np
import cv2
from matplotlib import pyplot as plt
#import tensorflow as tf
#import pprint
import multiprocessing as mp

im_dir = '/home/charlie/kaggle_sample/'
label_dir = '/home/charlie/kaggle_sample/stage1_labels.csv'

class DicomDict:
    ''' DicomDict
    DicomDict constructs the dictionary of all DICOM images,
    their labels, collects useful metrics, and does basic error
    checking

    Args:
        path_to_images: Str of valid path to folder containing top level folders
        path_to_labels: Str of valid path to csv file containing ids and labels

    Retruns:
        DicomDict.<args>
        DicomDict.dicom_dict: Dictionary of the following structure

            {top level scan id:
            ---{'cancer':label}
            ---{'slice_count':count of slices in scan}
            ---{'size':file size in bytes}
            }

        DicomDict.total_size: Total size in bytes of all DICOM images
        DicomDict.job_args: A list of tuples of the following structure

            [(path_to_image_slice, label, top level scan id),(...)]

        DicomDict.batch_size_limit:  An Int of recommended limit to batch size 
            based on total memory available on system and total_size taking 
            into account the necessary estimated overhead to perform batch 
            processing
    '''
    def __init__(self, path_to_images, path_to_labels):
        self.path_to_images = self.__check_path(path_to_images)
        self.path_to_labels = self.__check_path(path_to_labels)
        self.dicom_dict = self.__build_dict()
        self.total_size = self.__total_size()
        self.job_args = [(os.path.join(self.path_to_images,k), 
            list(j for i, j in v.items() if i == 'cancer')[0], 
            k) for k, v in self.dicom_dict.items()]
        self.batch_size_limit = self.__batch_limiter()
        
    def __check_path(self, path):
        if os.path.isfile(path):
            return path
        else:
            if os.path.isdir(path):
                return path
            else:
                raise ValueError('There was nothing in %s' % path)


    def __batch_limiter(self):
        mem = psutil.virtual_memory()
        tot = self.total_size
        return tot//mem.total

    def __total_size(self):
        total_size = 0
        for s in list(self.dicom_dict.values()):
            total_size += s['size']
        return total_size

    def __build_dict(self):
        print('Traversing folders')
        dicom_dict = dict()
        try:
            for root, dirs, files in tqdm(os.walk(self.path_to_images)):
                path = root.split(os.sep)
                top_id = os.path.basename(root)
                if len(files) > 0:
                    dicom_dict[top_id] = {}
                    dicom_dict[top_id]['slice_count'] = len(files)
                    dicom_dict[top_id]['size'] = 0
                    for f in files:
                        try:
                            dicom_dict[top_id]['size'] += os.path.getsize(os.path.join(root, f))
                        except FileNotFoundError:
                            continue
        except:
            raise ValueError('Dictionary Failed')
        return self.__assign_labels(dicom_dict)

    def __assign_labels(self,dicom_dict):
        print('Assigning labels')
        try:
            with open(self.path_to_labels) as cf:
                try:
                    reader = csv.DictReader(cf,delimiter=',')
                except:
                    raise FileNotFoundError('label file not found')

                for row in reader:
                    try:
                        dicom_dict['%s' % row['id']]['cancer'] = row['cancer']
                    except:
                        print('%s image does not exist' % row['id'])
        except KeyError:
            pass
        except:
            raise ValueError('Labels not assigned')

        return self.__filtered_dict(dicom_dict)

    def __filtered_dict(self, dicom_dict):
        del_count = 0
        for x in tqdm(list(dicom_dict.keys())):
            for y in list(dicom_dict[x].keys()):
                if x in dicom_dict.keys() and 'cancer' not in dicom_dict[x].keys():
                    #print(x)
                    #print(dicom_dict[x].keys())
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
    '''DicomImage 
    This object stores the id, label, and an np.array of the CT scan.

    Args:
        path_to_image: String of path to top folder storing slices
        label: Int of class label
        im_id: name of top folder storing slices

    Returns:
        An object containing:

        DicomImage.scan: np.array of scan
        DicomImage.spacing: 3x1 array of spacing adjustment on image
        DicomImage.<Args>

    '''
    def __init__(self, path_to_image, label, im_id):
        self.path_to_image = path_to_image
        self.label = label
        self.im_id = im_id
        self.scan = self.__load_scan()
        self.rescale_flag = False
        self.resample_flag = False
        self.image = []
        self.spacing = []
        

    def preview(self):
        first_patient_pixels = self.image[0]
        plt.imshow(first_patient_pixels[80], cmap=plt.cm.gray)
        plt.show()

    def __masking(self):
        '''do something with masking'''

        
    def resample(self, new_spacing=[1,1,1]):
        print(self.spacing)
        assert(self.rescale_flag is True, 'Need to rescale first!')
        im_rescaled = self.image[0]
        spacing = np.array([self.scan[0].SliceThickness] + self.scan[0].PixelSpacing, dtype=np.float32)
        resize_factor = spacing / new_spacing
        new_real_shape = im_rescaled.shape * resize_factor
        new_shape = np.round(new_real_shape)
        real_resize_factor = new_shape / im_rescaled.shape
        new_spacing = spacing / real_resize_factor
        image = scipy.ndimage.interpolation.zoom(self.im_rescaled, real_resize_factor, mode='nearest') 
        self.spacing.append(new_spacing)
        self.image.append(image)
        self.resample_flag = True

    def rescale(self):
        image = np.stack([s.pixel_array for s in self.scan])
        image = image.astype(np.int16)
        image[image == -2000] = 0
        for i in tqdm(range(len(self.scan))):
            intercept = self.scan[i].RescaleIntercept
            slope = self.scan[i].RescaleSlope
            if slope != 1:
                image[i] = slope * image[i].astype(np.float64)
                image[i] = image[i].astype(np.int16)
            image[i] += np.int16(intercept)
        self.image.append(np.array(image, dtype=np.int16))
        self.rescale_flag = True

    def __load_scan(self):
        slices = [dicom.read_file(self.path_to_image + '/' + s) for s in os.listdir(self.path_to_image)]
        slices.sort(key = lambda x: int(x.ImagePositionPatient[2]))
        try:
            slice_thickness = np.abs(slices[0].ImagePositionPatient[2]-slices[1].ImagePositionPatient[2])
        except:
            slice_thickness = np.abs(slices[0].SliceLocation - slices[1].SliceLocation)
        for s in slices:
            s.SliceThickness = slice_thickness
        return slices

def batch_rescale(imobj):
    return imobj.rescale()

def batch_resample(imobj):
    return imobj.resample()

class DicomBatch:
    def __init__(self, dicomDict):
        self.job_args = dicomDict.job_args
        self.total_samples = len(self.job_args)
        self.all_dicomImages = self.__load_batch_of_dicomImages()

    def __dicom_images(self, job_args):
        return DicomImage(*job_args)


    def __load_batch_of_dicomImages(self):
        im_batch = []

        print('Loading image files into memory')
        for k in tqdm(self.job_args):
                im_batch.append(self.__dicom_images(k))

        print('Processing images')
        with mp.Pool() as p:
            print(p.map(batch_rescale, im_batch))
        return im_batch


def main(argv=None):
    x = DicomDict(im_dir, label_dir)
    print(x.batch_size_limit)
    y = DicomBatch(x)
    print(y.all_dicomImages[0].image)

if __name__ == '__main__':
   main() 
