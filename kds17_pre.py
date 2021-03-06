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
import psutil
import dicom
import csv
import os
from scipy import ndimage as nd
from tqdm import tqdm
import numpy as np
from skimage import measure, morphology
import threading
import logging

logging.basicConfig(level=logging.DEBUG, format='[%(threadName)-9s] %(message)s',) '''Logging Format Specification command'''

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
    def __init__(self, path_to_image, label, im_id): '''Initialization method/Constructor containing arguments self, path_to_image, label, im_id'''
        self.path_to_image = path_to_image '''Assigns the path_to_image constructor argument to the self.path_to_image variable'''
        self.label = label '''Assigns the label constructor argument to the self.label variable'''
        self.im_id = im_id '''Assigns the im_id constructor argument to the self.im_id variable'''
        self.image, self.spacing = self.__load_scan() '''The values returned from the self.__load_scan() function are assigned to the self.image and the self.spacing 
variables'''

    def __rescale(self, slices): '''Class method __rescale with arguments self and slices'''
        image = np.stack([s.pixel_array for s in slices]) '''Concatenates the s.pixel arrays and stores result under the variable image'''
        image = image.astype(np.int16) '''.astype() numpy method converts numpy int16 to an array and stores it under the variable image.'''
        image[image == -2000] = 0 '''Assigns 0 to images/pixels that are out of scan.'''
        for i in range(len(slices)): '''For Loop to execute command over a number of times equal to the number of elements in slices'''
            intercept = slices[i].RescaleIntercept '''The value b [in the relationship between stored values (SV) in Pixel Data and the output units {Output units = m*SV + b}]
for each slice is assigned a variable intercept'''
            slope = slices[i].RescaleSlope '''The value m [in the relationship between stored values (SV) in Pixel Data and the output units {Output units = m*SV + b}]
for each slice is assigned a variable slope'''
            if slope != 1:
                image[i] = slope * image[i].astype(np.float64)
                image[i] = image[i].astype(np.int16)
            image[i] += np.int16(intercept)
        self.rescale_flag = True
        return image

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
        DicomBatch.<Args>
    '''

    def __init__(self, job_args, name):
        self.name = name
        self.job_args = job_args
        self.total_samples = len(self.job_args)
        self.batch = self.__load_batch_of_dicomImages()
        
    def __dicom_images(self, job_args):
        return DicomImage(*job_args)

    def __load_batch_of_dicomImages(self):
        im_batch = []
        print('Loading image files as DicomImage objects into DicomBatch')
        for k in tqdm(self.job_args):
            im_batch.append(self.__dicom_images(k))
        return im_batch

class ProcessDicomBatch(threading.Thread):
    def __init__(self,group=None, target=None, name=None, DicomImage=(), verbose=None):
        super().__init__(group=group, target=target, name=name)
        self.DicomImage = DicomImage
        return

    def run(self):
        logging.debug('Processing image')
        im = self.DicomImage
        im.image = self.__mask(np.array(im.image,  dtype=np.int16),fill_lung_structures=False)
        spacing = [1,1,1]
        resize_factor = im.spacing / spacing
        real_resize_factor = np.round(im.image.shape * resize_factor) / im.image.shape
        im.spacing = im.spacing / real_resize_factor
        im.image = nd.interpolation.zoom(im.image, real_resize_factor, mode='nearest') 

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
            return image
        else:
            return image*nd.binary_dilation(binary_image, iterations=4)
