# -*- coding: utf-8 -*-
#!/usr/bin/env python3

"""
geometry_maker
geo_maker
euclid
sampler
generator
artificial_sampler
sssample_generator
data_factory
sample_factory
euclid_geometry_factory
sample_producer
procedural
producer
"""
import numpy as np

#import scipy.misc
import numpy as np
from skimage.draw import line_aa

""" Pixel range: [0,255]  Size: WxH=size_shape """
def simple_triangles(size_shape, RGB_CHANNELS):
    W,H = 14,14
    those = []
    for i in range(10):
        img = np.zeros((W, H), dtype=np.uint8)
        rr, cc, val = line_aa(1, 1, 8, 4)
        img[rr, cc] = val * 255
        rr, cc, val = line_aa(0, 13, 13, 4)
        img[rr, cc] = val * 255

        rgbi = i % RGB_CHANNELS

        #rr, cc, val = line_aa(0+i, 13-i, 13-i, 4)
        #img[rr, cc] = val * 255

        # scipy.misc.imsave("out.png", img)
        img_rgb = np.repeat(img[:,:, None], RGB_CHANNELS, axis=2)
        img_rgb[:,:,rgbi] = 0

        """ Check sizes """
        assert np.prod(img_rgb[:,:,0].shape) == size_shape

        those.append(img_rgb.flatten())
    return those

    #img = np.random.randn(64, size_shape)
    #return img

    #label = None
    #RESET_FRESH = None
    #return img, label, RESET_FRESH
