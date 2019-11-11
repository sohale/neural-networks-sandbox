# extract_plans.py

import glob
import numpy as np
import scipy.misc
import imageio
import os
import math

def ext_plans():
    """
    This function ...
    """

    directory = './pix2pix-tensorflow/stan_training'
    input_image_file = './set1.jpg'

    os.makedirs(directory, exist_ok=True)

    # big_img = np.asarray(imageio.imread('./set1.jpg'))
    big_img  = imageio.imread(input_image_file)
    big_img = np.asarray(big_img)
    print(big_img.shape)  # (682, 1000, 3)
    nx, ny = 17,25 # 25, 17
    x0,y0, mx,my = 36,31, 682-644, 1000-969

    w,h = big_img.shape[0], big_img.shape[1]
    box = (x0,y0, w-mx, h-my)
    print('box', box)
    dx,dy = (box[2]-box[0])/nx, (box[3]-box[1])/ny
    print('dx,dy', dx,dy)
    # dataset = []
    counter = 0
    for xi in range(3,nx): #range(3): #(nx):
        for yi in range(ny): #range(3): #(ny):
            lx, ly = (x0+dx*xi), (y0+dy*yi)
            #print('lx,ly', lx,ly)
            rect = math.floor(lx), math.floor(lx+dx), math.floor(ly), math.floor(ly+dy)
            #print('rect', rect)
            im = big_img[rect[0]:rect[1], rect[2]:rect[3], :]
            # im = big_img[10:49,49:88,:]

            """
            im_grey = np.mean(im,axis=2)[:,:,None]
            WHITE_D = 80 # 50  10
            BLACK_D = 40 #100
            im = 255 - im
            none_white = im_grey < (255 - WHITE_D)
            im = im * none_white
            im = 255 - im
            im = im * (im_grey > BLACK_D)
            # im = im * (im_grey < 255-WHITE_D)
            #im = 255 - im
            """
            im_grey = np.mean(im,axis=2)[:,:,None]
            WHITE_D = 40 #80 # 50  10
            im = 255 - im
            none_white = im_grey < (255 - WHITE_D)
            im = im * none_white
            im = 255 - im
            im_mask = 255 - np.repeat(none_white.astype(im.dtype) * 255, 3, axis=2)

            print(im.shape, im_mask.shape)
            im_pair = np.concatenate( (im_mask, im), axis=1)

            #print('shp',big_img.shape)
            #print('saving', im.shape)

            # dataset.append(im)
            #filename = os.path.join('output','p'+str(xi)+'-'+str(yi)+'.png')
            counter += 1
            filename = os.path.join(directory, 'item'+str(counter)+'.png')
            imageio.imwrite(filename,  im_pair )
            #print('saved')

    #pict_array2d = np.asarray(img)
    #pict_array2d = scipy.misc.imresize(pict_array2d, (200,200))


train_plans = ext_plans()

"""
# Installation: (Windows)
# Install latest anaconda 64 bit
# Go to Anaconda commandline
conda create --name tensorf python=3.5
conda activate tensorf
pip install cython
pip install PyHamcrest
python -m pip install -U matplotlib
conda install scipy
conda install -c menpo imageio
pip install tensorflow==1.15.0



# First time only:
# From git
git clone git@github.com:sosi-org/neural-networks-sandbox.git

# Produce images training data
cd dataset-from-stan
python extract_plans_from_grid.py
cd pix2pix-tensorflow
# Starts training
python pix2pix.py  --mode train --output_dir stan_out   --input_dir stan_training
# Wait...

MacOS:
Installation on MacOS: (First time only)
    * virtualenv --version # If error, install virsualenv . see https://www.tensorflow.org/install/pip
    * cd dataset-from-stan
    * virtualenv -v --python=python3  ./tensorf
    * pip install tensorflow==1.15.0
    * pip install scipy
    * pip install imageio
    
    Unsure: cython PyHamcrest

Run on MacOS
    * cd dataset-from-stan
    * source ./tensorf/bin/activate
    * python extract_plans_from_grid.py

"""
