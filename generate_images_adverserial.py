#!/usr/bin/env python

import numpy as np
from os import getcwd
from os.path import join

# data path
path = join(getcwd(), "data/cuda")

def save(f, fname):
    img_size, img = f()
    img = np.insert(img, 0, img_size)
    fname = join(path, fname + '-{}'.format(img_size))
    print 'Saving {}'.format(fname)
    np.savetxt(fname, img, fmt="%d", delimiter="\n")

# generate adverserial images

def zeros():
    img_size = 10000000
    return img_size, np.zeros(img_size)

save(zeros, "zeros")
