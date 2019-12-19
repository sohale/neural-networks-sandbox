
"""
image loader for tensorflow 

Usage:

main_artworks_list = load_main_images(img, label)
batch = choose_random_batch(main_artworks_list, batch_size_provisional)

"""

import glob
import numpy as np
import scipy.misc
import imageio

# if False:
def load_main_images_0():
    """
    This function ...
    """


    arts = []
    for image_path in glob.glob("./art1/lines/*.jpg"):
            img = imageio.imread(image_path)
            pict_array2d = np.asarray(img)
            #pict_array2d = scipy.misc.imresize(pict_array2d, FIXED_SIZE[0:2])
            pict_array2d = imageio.imresize(pict_array2d, FIXED_SIZE[0:2])


            # normalise:
            #pict_array2d = np.mean(pict_array2d, axis=2)
            #pict_array2d = 1.0 - (pict_array2d / 255.0)
            pict_array2d = (pict_array2d / 255.0)
            #p1 = np.sum(pict_array2d.ravel()>1.0)
            #if (p1>0): print('>1.0', p1)
            #p0 = np.sum(pict_array2d.ravel()<0.0)
            #if (p0>0): print('<0.0', p0)

            assert FIXED_SIZE == pict_array2d.shape

            arts.append( pict_array2d )

    return arts


def load_main_images(img, label):
    arts = []
    for i in range(90):
            #print(i)
            pict_array2d = loadmnist2(img, label, i)
            pict_array2d = scipy.misc.imresize(pict_array2d, FIXED_SIZE[0:2])


            S, V0 = 1.0, 0.0

            pict_array2d = pict_array2d.astype(float) * S + V0
            
            #pict_array2d = (pict_array2d / 255.0)

            assert FIXED_SIZE == pict_array2d.shape

            arts.append( pict_array2d.ravel() )

    return arts


def choose_random_batch(main_artworks_list, batch_size_provisional, FLATTENED_SIZE, RGB_CHANNELS, avoid_resampling):
    # Select random parts of the main 4 artworks
    #return main_artworks_list
    batch = []
    if avoid_resampling:
        for i in range(batch_size_provisional):
            ii = np.random.randint(0,len(main_artworks_list))
            image = main_artworks_list[ii]
            assert (FLATTENED_SIZE,) == image.shape
            batch.append( image )
        return batch

    for i in range(batch_size_provisional):
        ii = np.random.randint(0,len(main_artworks_list))

        image = main_artworks_list[ii]

        image = image.copy()

        #MAX_VAL = 0.9
        MAX_VAL = 1.0
        #S, V0 = 1.0 / 255.0 * MAX_VAL,  0.0 * 0.9
        S, V0 = 1.0 / 255.0 * MAX_VAL,  0.0
        image = image * S + V0
        image[0] = 0.0  # -0.98

        if image is not None:
            batch.append( image )

    # test size andd pixel value range
    for img in batch:
        print('*min max:', np.min(img.ravel()), np.max(img.ravel()))

        p1 = np.sum(img.ravel()>1.0)
        if (p1>0): print('>1.0', p1)
        p0 = np.sum(img.ravel()<0.0)
        if (p0>0): print('<0.0', p0)
        print(FLATTENED_SIZE, img.shape)
        assert (FLATTENED_SIZE,) == img.shape
        print(img.shape)
    print('batch size', len(batch))

    return batch
