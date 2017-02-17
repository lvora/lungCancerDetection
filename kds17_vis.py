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

"""Visualisation handling of DICOM Images for Kaggle Data Science 2017

"""

def animate(image):
    fig = plt.figure()
    ims = []
    ax = fig.add_subplot(111)
    for i in range(image.shape[0]):
        im = plt.imshow(image[i], cmap=plt.cm.gray)
        ims.append([im])
    ani = an.ArtistAnimation(fig, ims, interval=150, blit=True)
    plt.show()

def rotate(image, pos):
    pos_dict = {'anterposterior' : (1,0),
                'r2l_saggital' : (2,0)}
    if pos == 'transverse':
        return image
    else:
        return np.rot90(image,axes=pos_dict[pos])

