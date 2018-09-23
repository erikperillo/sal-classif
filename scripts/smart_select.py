#!/usr/bin/env python3

import skimage.io as io
import skimage.morphology as morph
import skimage.measure as meas
import skimage.transform as transf
import multiprocessing as mp
import math
import numpy as np
import random
import os
import shutil
import glob

IMG_SRC_DIR = '/home/erik/experiments/sal-classif/data/imagenet-raw'
SAL_SRC_DIR = '/home/erik/experiments/sal-classif/data/imagenet-sals'
HARD_FOCUSING_DIR = \
    '/home/erik/experiments/sal-classif/data/imagenet-hard-focus'
SOFT_FOCUSING_DIR = \
    '/home/erik/experiments/sal-classif/data/imagenet-soft-focus'
RAND_FOCUSING_DIR = \
    '/home/erik/experiments/sal-classif/data/imagenet-rand-focus'
FOCUS_SHAPE = (299, 299)

def disk(radius):
    return morph.disk(radius)

THR = 0.333
MORPH_OPS = [
    ('opening', disk(3)),
    ('closing', disk(5)),
]

def morph_op(mask, ops):
    for op, selem in ops:
        if op == "dilation":
            mask = morph.binary_dilation(mask, selem=selem).astype("uint8")
        elif op == "erosion":
            mask = morph.binary_erosion(mask, selem=selem).astype("uint8")
        elif op == "opening":
            mask = morph.binary_opening(mask, selem=selem).astype("uint8")
        elif op == "closing":
            mask = morph.binary_closing(mask, selem=selem).astype("uint8")
        else:
            raise ValueError("unknown morph op '{}'".format(op))
    return mask

def load_img(path):
    return io.imread(path)

def unit_norm(x):
    return (x - x.min())/(x.max() - x.min())

def load_sal(path):
    sal = load_img(path)
    sal = sal/255
    return sal

def save_img(path, img):
    return io.imsave(path, img, quality=100)

def select_point_hard(sal, thr=THR, morph_ops=MORPH_OPS):
    mask = (sal >= thr).astype('uint8')
    mask = morph_op(mask, morph_ops)
    labels = meas.label(mask)
    regionprops = meas.regionprops(labels)
    if not regionprops:
        print('warning: no props')
        return sal.shape[0]//2, sal.shape[1]//2
    prop = max(regionprops, key=lambda p: p.area)
    row, col = prop.centroid
    row, col = int(row), int(col)
    return row, col

def select_point_soft(sal, power=2):
    height, width = sal.shape[:2]
    x_mesh, y_mesh = np.meshgrid(np.arange(width), np.arange(height))
    sal = sal**power
    sal = sal/sal.sum()
    x_mean = (sal*x_mesh).sum()
    y_mean = (sal*y_mesh).sum()
    x_mean, y_mean = int(x_mean), int(y_mean)
    return y_mean, x_mean

def select_point_rand(sal):
    height, width = sal.shape[:2]
    row = random.randint(0, height-1)
    col = random.randint(0, width-1)
    return row, col

def focus(img, coord, shape):
    row, col = coord
    h, w = shape
    img_h, img_w = img.shape[:2]
    assert img_h >= h
    assert img_w >= w

    min_i = max(row - math.floor(h/2), 0)
    if min_i + h > img_h:
        min_i -= (min_i + h) - img_h
    min_j = max(col - math.floor(w/2), 0)
    if min_j + w > img_w:
        min_j -= (min_j + w) - img_w

    focused = img[min_i:min_i+h, min_j:min_j+w]
    assert focused.shape[:2] == shape
    return focused

def get_sal_path(img_path):
    filename = os.path.basename(img_path).replace('.JPEG', '.png')
    path = os.path.join(SAL_SRC_DIR, filename)
    return path

def select_focus_save(path):
    print('on file', path)
    img = load_img(path)
    sal = load_sal(get_sal_path(path))
    if sal.shape[:2] != img.shape[:2]:
        print('resizing')
        sal = transf.resize(sal, img.shape[:2], mode='constant')
        sal = unit_norm(sal)

    row, col = select_point_soft(sal)
    focus_img = focus(img, (row, col), FOCUS_SHAPE)
    dst_path = os.path.join(SOFT_FOCUSING_DIR, os.path.basename(path))
    save_img(dst_path, focus_img)

    row, col = select_point_hard(sal)
    focus_img = focus(img, (row, col), FOCUS_SHAPE)
    dst_path = os.path.join(HARD_FOCUSING_DIR, os.path.basename(path))
    save_img(dst_path, focus_img)

    row, col = select_point_rand(sal)
    focus_img = focus(img, (row, col), FOCUS_SHAPE)
    dst_path = os.path.join(RAND_FOCUSING_DIR, os.path.basename(path))
    save_img(dst_path, focus_img)

def main():
    for p in [HARD_FOCUSING_DIR, SOFT_FOCUSING_DIR, RAND_FOCUSING_DIR]:
        if os.path.isdir(p):
            shutil.rmtree(p)
        os.makedirs(p)

    paths = glob.glob(os.path.join(IMG_SRC_DIR, '*'))
    print(len(paths), 'paths')

    pool = mp.Pool(8)
    pool.map(select_focus_save, paths)

if __name__ == '__main__':
    main()
