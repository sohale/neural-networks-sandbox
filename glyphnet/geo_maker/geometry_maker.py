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


EPS = 0.0000000001

# point generators
def on_border_square(c):
    corners = [(0.,0.), (1.,0.), (1.,1.), (0.,1.)]
    corners.append(corners[0]) # close it
    N_sides = len(corners) - 1

    # c = c % N_sides
    if c > 1 + EPS:
        raise
    c = c % 1.0

    ce = c * N_sides
    alpha = ce % 1.0

    #if ce >= N_sides:
    #    ce = N_sides
    #    alpha = 1.0

    cs = int(np.floor(ce / 1.0)) # segment
    if cs == N_sides:
        cs = N_sides-1
    begin = corners[cs]
    end = corners[cs+1]
    X_, Y_  = 0,1
    return (
        begin[X_]*(1. - alpha) + end[X_] * alpha,
        begin[Y_]*(1. - alpha) + end[Y_] * alpha,
    )

def test():
    expected = [(0.0, 0.0), (0.5, 0.0), (1.0, 0.5), (1.0, 1.0), (0.0, 1.0), (0.0, 1.0)]
    l = []; ctr = 0
    for c in [0.0, 0.25/2, 0.25+0.25/2, 0.5, 0.75, 1.0]:
        print(c, '->', on_border_square(c), expected[ctr])
        ctr += 1
        # expected
        l.append(on_border_square(c))
    print(repr(l))
test()
# exit()

""" Pixel range: [0,255]  Size: WxH=size_shape """
def simple_triangles(size_shape, RGB_CHANNELS, how_many_samples=10):
    W,H = 14,14
    those = []
    for i in range(how_many_samples):
        # img_rgb = np.zeros((W,H,RGB_CHANNELS), dtype=np.uint8)
        img_rgb = np.zeros((W,H,RGB_CHANNELS), dtype=float)

        rgbi = i % RGB_CHANNELS

        # img = np.zeros((W, H), dtype=np.uint8)
        # img = np.zeros((W, H), dtype=float)
        if False:
            rr, cc, val = line_aa(1, 1, 8, 4)
            img_rgb[rr, cc, rgbi] = val * 255
            rr, cc, val = line_aa(0, 13, 13, 4)
            img_rgb[rr, cc, rgbi] = val * 255

            rr, cc, val = line_aa((0+i)%W, (13-i)%H, (13-i)%W, 4 % H)
            img_rgb[rr, cc, rgbi] += val * 255

        category = 'sq-border'

        if category == 'sq-border':
            x1, y1 = on_border_square(np.random.rand())
            x2, y2 = on_border_square(np.random.rand())
            x1 = int(x1 * (W-1))
            y1 = int(y1 * (H-1))
            x2 = int(x2 * (W-1))
            y2 = int(y2 * (H-1))
        elif category == 'limp1':
            x1,y1, x2,y2 = (0+i)%W, (13-i)%H, (13-i)%W, 4
        else:
            raise Error('bad category')

        #rr, cc, val = line_aa(x1%W, y1%H, x2%W, y2 % H)
        rr, cc, val = line_aa(x1, y1, x2, y2 )
        print('drawing:', x1, y1, x2, y2 )
        img_rgb[rr, cc, rgbi] = np.maximum(img_rgb[rr, cc, rgbi], val * 255)

        img_rgb[img_rgb>255.0] = 255.0
        # img_rgb = img_rgb / 2.0

        # scipy.misc.imsave("out.png", img)
        #img_rgb_delta = np.repeat(img[:,:, None], RGB_CHANNELS, axis=2)
        #if RGB_CHANNELS > 1:
        #    img_rgb[:,:,rgbi] = 0

        # img_rgb[:,:, rgbi] += img

        """ Check sizes """
        assert np.prod(img_rgb[:,:,0].shape) == size_shape

        f = img_rgb.flatten().astype(float) / 255.0
        # print('max:', np.max(f))
        those.append(f)
    return those

    #img = np.random.randn(64, size_shape)
    #return img

    #label = None
    #RESET_FRESH = None
    #return img, label, RESET_FRESH
