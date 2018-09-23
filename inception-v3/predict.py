#!/usr/bin/env python3

#import tensorflow_hub as hub
import json
#import tensorflow as tf
import multiprocessing as mp
import numpy as np
from skimage import io
import glob
import os
from keras.applications import inception_v3

IMGS_DIR = '/home/erik/experiments/sal-classif/data/imagenet-hard-focus'
# inception-v3
#MODULE_URL = 'https://tfhub.dev/google/imagenet/inception_v3/classification/1'

def load_img(path):
    return io.imread(path)

def partition(lst, size):
    parts = []
    for i in range(0, len(lst), size):
        parts.append(lst[i:i+size])
    return parts

def pre_proc(batch):
    return inception_v3.preprocess_input(batch)

def _mk_batch(imgs):
    if not isinstance(imgs, (list, tuple)):
        imgs = [imgs]
    return np.stack(imgs, axis=0)

def mk_batch(paths):
    imgs = [load_img(p) for p in paths]
    batch = _mk_batch(imgs)
    batch = pre_proc(batch)
    return batch

def main():
    paths = glob.glob(os.path.join(IMGS_DIR, '*'))
    paths_parts = partition(paths, 128)

    pool = mp.Pool(8)

    batches = pool.map(mk_batch, paths_parts)

    model = inception_v3.InceptionV3()

    dct = {}
    for paths, batch in zip(paths_parts, batches):
        print('in paths {} ...'.format(paths[0]))
        preds = model.predict(batch)
        preds = [np.argmax(p) for p in preds]
        for p, y in zip(paths, preds):
            dct[p] = int(y)
    print('bla')

    with open('preds.json', 'w') as f:
        json.dump(dct, f, indent=4, sort_keys=True)
    print('ble')

if __name__ == '__main__':
    main()
