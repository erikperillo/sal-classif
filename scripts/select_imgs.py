#!/usr/bin/env python3

import multiprocessing as mp
from skimage import io
import math
import shutil
import random
import os
import glob

'''
this script:
    - samples images from imagenet directory;
    - copies selected images to another dir;
    - copies a centered version of images to yet another dir;
    - makes a list of selected sources.
'''

SHAPE = (299, 299)
RAW_SAVE_DIR = '/home/erik/experiments/sal-classif/data/imagenet-raw'
CENTERED_SAVE_DIR = \
    '/home/erik/experiments/sal-classif/data/imagenet-central-focus'
SRC_IMGS_DIR = '/home/erik/data/imagenet-2012-val/images'
SRC_PATHS_LIST_PATH = \
    '/home/erik/experiments/sal-classif/data/imagenet-srcs.csv'
N_TRIALS = 50000

def load(path):
    return io.imread(path)

def pre_proc(img):
    h, w, __ = img.shape
    h_len, w_len = SHAPE
    h_slice = slice(h//2 - math.floor(h_len/2), h//2 + math.ceil(h_len/2))
    w_slice = slice(w//2 - math.floor(w_len/2), w//2 + math.ceil(w_len/2))
    img = img[h_slice, w_slice]
    print(img.shape)
    if img.shape[:2] != SHAPE:
        raise ValueError
    return img

def save(img, orig_path, base_dir):
    path = os.path.join(base_dir, os.path.basename(orig_path))
    io.imsave(path, img, quality=100)
    return path

def mv_image(path):
    print('in path {}:'.format(path))
    img = load(path)
    orig_img = img.copy()
    old_shape = img.shape
    try:
        img = pre_proc(img)
        print('\tfrom shape {} to {}'.format(old_shape, img.shape))
    except ValueError:
        print('\tignoring path, invalid shape {}'.format(img.shape))
        return None
    raw_img_path = save(orig_img, path, RAW_SAVE_DIR)
    centered_img_path = save(img, path, CENTERED_SAVE_DIR)
    print('\t saved to', centered_img_path)
    return centered_img_path

def main():
    for path in [RAW_SAVE_DIR, CENTERED_SAVE_DIR]:
        if os.path.isdir(path):
            shutil.rmtree(path)
        os.makedirs(path)

    paths = glob.glob(os.path.join(SRC_IMGS_DIR, '*'))
    random.shuffle(paths)
    paths = paths[:N_TRIALS]

    pool = mp.Pool(8)
    paths = pool.map(mv_image, paths)
    paths = [p for p in paths if p is not None]
    print('{} good paths'.format(len(paths)))

    with open(SRC_PATHS_LIST_PATH, 'w') as f:
        for p in paths:
            print(p, file=f)
    print('saved list of src paths to', SRC_PATHS_LIST_PATH)

if __name__ == '__main__':
    main()
