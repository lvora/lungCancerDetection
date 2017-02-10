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
from scipy import ndimage as nd
import time
from tqdm import tqdm
import numpy as np
import cv2
import threading as th
import pickle
from matplotlib import pyplot as plt
from matplotlib import animation as an
from skimage import measure, morphology
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

import kds17_io as kio
#import tensorflow as tf
#import pprint
#import multiprocessing as mp

im_dir = '/home/charlie/kaggle_stage1/'
label_dir = '/home/charlie/kaggle_stage1/stage1_labels.csv'
pickle_dir = '/home/charlie/kaggle_pickles'

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
        self.batched_job_args = [self.job_args[i:i+self.batch_size_limit] for i in range(0,len(self.job_args), self.batch_size_limit)]
        
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
        if (20*tot//mem.total) < 1:
            factor = 1
        else:
            factor = (20*tot//mem.total)
        limit = len(self.job_args)//factor
        print('Batch size limit set to %i' % limit)
        return limit

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
                        pass
                        #print('%s image does not exist' % row['id'])
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
        DicomImage.spacing: 3x1 array of spacing adjustment on 
                            image to normalize to 1mm^3
        DicomImage.<Args>

    '''
    def __init__(self, path_to_image, label, im_id):
        self.path_to_image = path_to_image
        self.label = label
        self.im_id = im_id
        self.image, self.spacing = self.__load_scan()

    def animate(self):
        fig = plt.figure()
        fig.suptitle('%s with label %s' % (self.im_id, self.label))
        ims = []
        ax = fig.add_subplot(111)
        for i in range(self.image.shape[0]):
            im = plt.imshow(self.image[i], cmap=plt.cm.plasma)
            ims.append([im])
        ani = an.ArtistAnimation(fig, ims, interval=50, blit=True)
        plt.show()

    def __largest_label_volume(self,im, bg=-1):
        vals, counts = np.unique(im, return_counts=True)
        counts = counts[vals != bg]
        vals = vals[vals != bg]
        if len(counts) > 0:
            return vals[np.argmax(counts)]
        else:
            return None

    def __mask(self,image, fill_lung_structures=True):
        binary_image = np.array(image > -320, dtype=np.int8)+1
        binary_image_coarse = np.array(image[image>-320], dtype=np.int8)+1-image
        labels = measure.label(binary_image)
        bbr = np.array(labels.shape)-1
        top = labels[bbr[0],bbr[1],bbr[2]]
        mid = labels[0,bbr[1],bbr[2]]
        bot = labels[0,0,0]

        background_label = (top+mid+bot)/3

        binary_image[background_label == labels] = 2

        if fill_lung_structures:
            for i, axial_slice in enumerate(binary_image):
                axial_slice = axial_slice - 1
                labeling = measure.label(axial_slice)
                l_max = self.__largest_label_volume(labeling, bg=0)

                if l_max is not None: #This slice contains some lung
                    binary_image[i][labeling != l_max] = 1

        binary_image -= 1 #Make the image actual binary
        binary_image = 1-binary_image # Invert it, lungs are now 1

        labels = measure.label(binary_image, background=0)
        l_max = self.__largest_label_volume(labels, bg=0)
        if l_max is not None: # There are air pockets
            binary_image[labels != l_max] = 0
        if binary_image[np.nonzero(binary_image)].shape[0] < 2000000:
            return image*nd.binary_dilation(binary_image_coarse, iterations=4)
        else:
            return image*nd.binary_dilation(binary_image, iterations=4)


    def __rescale(self, slices):
        image = np.stack([s.pixel_array for s in slices])
        image = image.astype(np.int16)
        image[image == -2000] = 0
        for i in range(len(slices)):
            intercept = slices[i].RescaleIntercept
            slope = slices[i].RescaleSlope
            if slope != 1:
                image[i] = slope * image[i].astype(np.float64)
                image[i] = image[i].astype(np.int16)
            image[i] += np.int16(intercept)
        self.rescale_flag = True
        #return self.__mask(np.array(image, dtype=np.int16))
        return self.__mask(np.array(image,  dtype=np.int16),fill_lung_structures=False)

    def __load_scan(self):
        slices = [dicom.read_file(self.path_to_image + '/' + s) for s in os.listdir(self.path_to_image)]
        slices.sort(key = lambda x: int(x.ImagePositionPatient[2]))
        try:
            slice_thickness = np.abs(slices[0].ImagePositionPatient[2]-slices[1].ImagePositionPatient[2])
        except:
            slice_thickness = np.abs(slices[0].SliceLocation - slices[1].SliceLocation)
        for s in slices:
            s.SliceThickness = slice_thickness

        spacing = np.array([slices[0].SliceThickness] +  slices[0].PixelSpacing, dtype=np.float32)
        return self.__rescale(slices), spacing

class DicomBatch:
    '''DicomBatch 
    This object stores a list of DicomImage objects and performs batch processing

    Args:
        DicomDict:
        name:

    Methods:
        DicomBatch.process_batch(f):

    Returns:
        An object containing:

        DicomBatch.job_args: 
        DicomBatch.total_samples: 
        DicomBatch.spacing: 
        DicomBatch.batch: 
        DicomBatch.process_list: 
        DicomBatch.<Args>

    '''
    def __init__(self, dicomDict, name):
        self.name = name
        self.job_args = dicomDict.job_args
        self.total_samples = len(self.job_args)
        self.spacing = [1,1,1]
        self.batch = self.__load_batch_of_dicomImages()
        self.process_list = []
        
    def __maybe_mkdir(self, path):
        if os.path.isdir(path):
            return path
        else:
            os.path.mkdir(path)
            return path

    def __dicom_images(self, job_args):
        return DicomImage(*job_args)

    def __load_batch_of_dicomImages(self):
        im_batch = []
        print('Loading image files as DicomImage objects into memory')
        for k in tqdm(self.job_args):
            im_batch.append(self.__dicom_images(k))
        return im_batch

    def process_batch(self,f):
        fun_dict = {'zoom':self.__resample}
        threads = []
        stop_event = th.Event()
        for im in self.batch:
            p = th.Thread(target=fun_dict[f], args=(im,))
            threads.append(p)
        now = time.strftime('%H:%M:%S',time.localtime())
        print('%s - Batch processing %i images with %s.' % (now,len(threads),f))
            
        try:
            [(x.start(), x.join()) for x in threads]
            if not any(x.is_alive() for x in threads):
                now = time.strftime('%H:%M:%S',time.localtime())
                self.process_list.append(f) 
                print('%s - Batch Processing complete' % now)
        except KeyboardInterrupt:
            stop_event.set()
            print('Interrupt Caught. Terminating now...')

    def __resample(self, im):
        resize_factor = im.spacing / self.spacing
        new_real_shape = im.image.shape * resize_factor
        new_shape = np.round(new_real_shape)
        real_resize_factor = new_shape / im.image.shape
        new_spacing = im.spacing / real_resize_factor
        start = time.time()
        im.image = nd.interpolation.zoom(im.image, real_resize_factor, mode='nearest') 
        fin = time.time()
        tim = fin-start
        now = time.strftime('%H:%M:%S',time.localtime())
        print('%s - Zoom complete in %.3f seconds '% (now,tim))
        im.spacing = new_spacing


def main(argv=None):
    #x = DicomDict(im_dir, label_dir)

    #y = DicomBatch(x, 'test_batch3')
    #y.process_batch('zoom')

    io = kio.DicomIO(pickle_dir)
    #io.save(y)

    z = io.load('test_batch3.pkl')

    z.batch[0].animate()
    
    print(z[0].batch[0].label)
    #print(z[1].batch[4].image.shape)


if __name__ == '__main__':
   main() 
