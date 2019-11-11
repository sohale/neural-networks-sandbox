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

    # original_img = np.asarray(imageio.imread('./set1.jpg'))
    original_img  = imageio.imread(input_image_file)
    original_img = np.asarray(original_img)
    print(original_img.shape)  # (682, 1000, 3)

    nyy, nxx = 17,25 # 25, 17
    # corners
    yy0,xx0, myy,mxx = 36,31, 38, 31

    box_main = (yy0,xx0, original_img.shape[0]-myy, original_img.shape[1]-mxx)
    print('box_main', box_main)

    dyy,dxx = (box_main[2]-box_main[0])/nyy, (box_main[3]-box_main[1])/nxx
    print('dyy,dxx', dyy,dxx)


    # dataset = []
    counter = 0
    for yyi in range(3,nyy): #range(3): #(nyy):
        for xxi in range(nxx): #range(3): #(nxx):

            # top-left for each sub-picture
            lyy, lxx = (yy0+dyy*yyi), (xx0+dxx*xxi)
            #print('lyy,lxx', lyy,lxx)
            subpic_rect = math.floor(lyy), math.floor(lyy+dyy), math.floor(lxx), math.floor(lxx+dxx)
            #print('subpic_rect', subpic_rect)
            im = original_img[subpic_rect[0]:subpic_rect[1], subpic_rect[2]:subpic_rect[3], :]
            # im = original_img[10:49,49:88,:]

            """
            im_grey = np.mean(im,axis=2)[:,:,None]
            WHITE_TOLERANCE = 80 # 50  10
            BLACK_TOLERANCE = 40 #100
            im = 255 - im
            none_white = im_grey < (255 - WHITE_TOLERANCE)
            im = im * none_white
            im = 255 - im
            im = im * (im_grey > BLACK_TOLERANCE)
            # im = im * (im_grey < 255-WHITE_TOLERANCE)
            #im = 255 - im
            """
            im_grey = np.mean(im,axis=2)[:,:,None]
            WHITE_TOLERANCE = 40 #80 # 50  10
            im = 255 - im
            none_white = im_grey < (255 - WHITE_TOLERANCE)
            im = im * none_white
            im = 255 - im
            im_mask = 255 - np.repeat(none_white.astype(im.dtype) * 255, 3, axis=2)

            print(im.shape, im_mask.shape)
            im_pair = np.concatenate( (im_mask, im), axis=1)

            #print('shp',original_img.shape)
            #print('saving', im.shape)

            # dataset.append(im)
            #filename = os.path.join('output','p'+str(yyi)+'-'+str(xxi)+'.png')
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
