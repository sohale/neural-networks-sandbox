
# -*- coding: utf-8 -*-
#!/usr/bin/env python3


import tensorflow as tf
import numpy as np
import time
import scipy.misc
import imageio


RGB3DIMS = 3


input = tf.placeholder(tf.uint8, [None, None, RGB3DIMS]) 
output = input * 2

data_im1 = np.ones([5,4,3], dtype=np.uint8)


sess = tf.Session()
out_data = \
    sess.run([output], feed_dict={input: data_im1})

print(out_data)
