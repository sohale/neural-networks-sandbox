# -*- coding: utf-8 -*-
#!/usr/bin/env python3

"""
Everything about mnist

Use python3
Usage:
```
img, label, RESET_FRESH = loadmnist_from_args()
```
"""


import random
import argparse
from mnist import MNIST



def loadmnist_from_args():

    parser = argparse.ArgumentParser()
    parser.add_argument("--id", default=None, type=int,
                        help="ID (position) of the letter to show")
    parser.add_argument("--training", action="store_true",
                        help="Use training set instead of testing set")

    parser.add_argument("--data", default="./data",
                        help="Path to MNIST data dir")


    parser.add_argument("--reset", default="NO",
                        help="reset the save")

    args = parser.parse_args()

    RESET_FRESH = not (args.reset == "NO")



    mn = MNIST(args.data)

    if args.training:
        img, label = mn.load_training()
    else:
        img, label = mn.load_testing()
    print("images", len(img), len(label))
    return  img, label, RESET_FRESH

def loadmnist2(img, label, which):
    img2d = np.array(img[which]).reshape(28,28)
    # S = (1.0/255.0) * 1.1
    img2d = img2d.astype(float)
    if False:
        img2d[img2d>1.0] = 1.0
    img2d = img2d[:,:, None]  # 28 x 28 x 1
    img2d = np.repeat(img2d, 3, axis=2)
    #print(img2d.shape)
    if False:
        d = which % 3
        if d == 0:
            img2d[:,:,1:3]=0
        elif d == 1:
            img2d[:,:,0:2]=0
        elif d == 2:
            img2d[:,:,0]=0
            img2d[:,:,2]=0

    #print(np.max(np.mean(img2d,axis=2), axis=0))
    #print(np.max(np.mean(img2d,axis=2), axis=1))


    print('***min max:', np.min(img2d.ravel()), np.max(img2d.ravel()))
    return img2d


