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

def no_conflicts_warp_32():
    factor = 312500
    upper = 32
    img_size = upper * factor # 10 million
    return img_size, np.tile(np.arange(upper), factor)

def no_conflicts_warp_256():
    factor = 39062
    upper = 256
    img_size = upper * factor # ~10 million
    return img_size, np.tile(np.arange(upper), factor)

save(zeros, "zeros")
save(no_conflicts_warp_32, "no-conflicts-warp-32")
save(no_conflicts_warp_256, "no-conflicts-warp-256")
