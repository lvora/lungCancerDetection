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
import kds17_pre as pre
import csv
import psutil
import os
from tqdm import tqdm
import pickle


class DicomDict:
    ''' DicomDict
    DicomDict constructs the dictionary of all DICOM images,
    their labels, collects useful metrics, and does basic error
    checking

    Args:
        path_to_images: Str of valid path to folder containing top 
                        level folders
        path_to_labels: Str of valid path to csv file containing ids 
                        and labels

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
        self.job_args = [(os.path.join(self.path_to_images, k), 
                         list(j for i, j in v.items() if i == 'cancer')[0], k) 
                         for k, v in self.dicom_dict.items()]
        self.batch_size_limit = self.__batch_limiter()
        self.batched_job_args = [self.job_args[i:i+self.batch_size_limit] 
                                 for i in range(0, len(self.job_args), 
                                 self.batch_size_limit)]
        
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
                top_id = os.path.basename(root)
                if len(files) > 0:
                    dicom_dict[top_id] = {}
                    dicom_dict[top_id]['slice_count'] = len(files)
                    dicom_dict[top_id]['size'] = 0
                    for f in files:
                        try:
                            dicom_dict[top_id]['size'] += os.path.getsize(
                                                         os.path.join(root, f))
                        except FileNotFoundError:
                            continue
        except:
            raise ValueError('Dictionary Failed')
        return self.__assign_labels(dicom_dict)

    def __assign_labels(self, dicom_dict):
        print('Assigning labels')
        try:
            with open(self.path_to_labels) as cf:
                try:
                    reader = csv.DictReader(cf, delimiter=',')
                except:
                    raise FileNotFoundError('label file not found')

                for row in reader:
                    try:
                        dicom_dict['%s' % row['id']]['cancer'] = row['cancer']
                    except:
                        pass
        except KeyError:
            pass
        except:
            raise ValueError('Labels not assigned')

        return self.__filtered_dict(dicom_dict)

    def __filtered_dict(self, dicom_dict):
        del_count = 0
        for x in list(dicom_dict.keys()):
            for y in list(dicom_dict[x].keys()):
                if (x in dicom_dict.keys() and 'cancer' 
                        not in dicom_dict[x].keys()):

                    del_count += 1
                    del dicom_dict[x]
        print(('Dictionary Built with %i items deleted for missing' 
               'labels or images') % del_count)
        return dicom_dict

    def save(self, path_to_save): 
        ''' DicomDict.save(path_to_save = STR)
        This dumps the dictionary to a csv file.  Only useful for 
        troubleshooting
        '''
        w = csv.writer(open(os.path.join(path_to_save, 
                            'dicom_dict_dump_kds17.csv'), 'w'))
        for key, val in self.dicom_dict.items():
            w.writerow([key, val])
        print('dicom_dict_dump_kds17.csv has been saved in %s' % path_to_save)


class DicomIO:
    ''' DicomIO
    DicomIO handles loading saving and checking of the dicom pickle files

    Args:
        pickle_dir: valid path to directory containing pickle files or
                    where they should be saved
    Retruns:
        DicomIO.<args>
        DicomIO.list: List containing the names of all files in pickle_dir

    '''
    def __init__(self, pickle_dir, im_dir=None, label_dir=None):
        self.label_dir = label_dir
        self.im_dir = im_dir
        self.pickle_dir = pickle_dir
        self.batch_file_list = self.__check_path(arg='batch')
        self.DicomDict = self.__load_dict()

    def __check_path(self, arg=None):
        if os.path.isdir(self.pickle_dir):
            if arg is None:
                return [f for f in os.listdir(self.pickle_dir) 
                        if os.path.isfile(os.path.join(self.pickle_dir, f))] 
            else:
                return [f for f in os.listdir(self.pickle_dir) if arg in f]
        else:
            os.mkdir(self.pickle_dir)
            self.__check_path()

    def save_dict(self):
        print('Saving dictionary.pkl in %s' % self.pickle_dir)
        with open(os.path.join(self.pickle_dir, 'dictionary.pkl',), 'wb') as f:
            pickle.dump(self.DicomDict, f)

    def __load_dict(self):
        if not self.__check_path():
            return DicomDict(self.im_dir, self.label_dir)
        else:
            if any(self.__check_path('dictionary')):
                print('Loading dictionary.pkl from %s' % (self.pickle_dir))
                with open(os.path.join(self.pickle_dir, 'dictionary.pkl'), 
                          'rb') as f:
                    return pickle.load(f)
            else:
                if self.im_dir is None or self.label_dir is None:
                    print('You need to assign an im_dir and label_dir')
                else:
                    print('Dead end??')

    def save_batch(self, dicomBatch): 
        print('Saving %s in %s' % (dicomBatch.name, self.pickle_dir))
        with open(os.path.join(self.pickle_dir, 
                               dicomBatch.name+'.pkl',), 'wb')as f:
            pickle.dump(dicomBatch, f)
        self.__check_path()
        
    def load_batch(self, pickle_name): 
        print('Loading %s from %s' % (pickle_name, self.pickle_dir))
        with open(os.path.join(self.pickle_dir, pickle_name), 'rb')as f:
            return pickle.load(f)

    def build_batch(self):
        print('Constructing batches of images and saving them in %s' 
              % self.pickle_dir) 
        for i, args in enumerate(tqdm(self.DicomDict.batched_job_args)):
            y = pre.DicomBatch(args, 'batch_%i' % i)
            threads = []
            for im in y.batch:
                t = pre.ProcessDicomBatch(DicomImage=im)
                t.setDaemon(True)
                threads.append(t)
            print('\nPreprocessing images for batch_%i\n' % i)
            [t.start() for t in threads]
            [t.join() for t in threads]
            self.save_batch(y)
        





        

