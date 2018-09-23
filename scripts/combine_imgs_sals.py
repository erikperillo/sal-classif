#!/usr/bin/env python3

import multiprocessing as mp
from skimage import io
import os
import numpy as np
import glob
import random

IMG_DIR = '/home/erik/experiments/sal-classif/data/imagenet-imgs'
SAL_DIR = '/home/erik/experiments/sal-classif/data/imagenet-sal-maps'
COMB_DIR = '/home/erik/experiments/sal-classif/data/imagenet-combs'

def unit_norm(x):
    return (x - x.min())/(x.max() - x.min())

def get_sal_path(img_path):
    filename = os.path.basename(img_path).replace('.jpeg', '.png')
    return os.path.join(SAL_DIR, filename)

def get_comb_path(img_path):
    filename = '{}-comb.png'.format(os.path.basename(img_path).rstrip('.jpeg'))
    return os.path.join(COMB_DIR, filename)

def combine(img_path):
    sal_path = get_sal_path(img_path)

    img = io.imread(img_path)
    sal = io.imread(sal_path)

    sal = unit_norm(sal)
    #sal = np.sqrt(sal)

    comb = img.copy().astype('float32')
    for i in range(3):
        comb[..., i] = (img[..., i].astype('float32'))*sal
    comb = comb.astype('uint8')

    dst_path = get_comb_path(img_path)
    print('{} + {} -> {}'.format(img_path, sal_path, dst_path))
    io.imsave(dst_path, comb, quality=100)

def main():
    if not os.path.isdir(COMB_DIR):
        os.makedirs(COMB_DIR)

    paths = glob.glob(os.path.join(IMG_DIR, '*.jpeg'))
    pool = mp.Pool(8)
    pool.map(combine, paths)

if __name__ == '__main__':
    main()
