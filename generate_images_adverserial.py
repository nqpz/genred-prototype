#!/usr/bin/env python

import numpy as np
from os import getcwd
import os.path

# data path
path = os.path.join(getcwd(), "data/cuda")

def save(f, fname):
    img_size, img = f()
    img = np.insert(img, 0, img_size)
    fname = os.path.join(path, fname + '-{}.dat'.format(img_size))
    if not os.path.exists(fname):
        print 'Saving {}'.format(fname)
        np.savetxt(fname, img, fmt="%d", delimiter="\n")

# generate adverserial images

def zeros():
    img_size = 10000000
    return img_size, np.zeros(img_size)

def no_conflicts_warp():
    factor = 312500
    img_size = 32 * factor # 10 million
    return img_size, np.tile(np.arange(32), factor)

save(zeros, "zeros")
save(no_conflicts_warp, "no-conflicts-warp")
