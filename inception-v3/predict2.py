#!/usr/bin/env python3

#import tensorflow_hub as hub
import json
#import tensorflow as tf
import multiprocessing as mp
import pandas as pd
import numpy as np
from skimage import io
from skimage import transform as transf
import glob
import os
from keras.applications import inception_v3

SHAPE = (299, 299)

IMGS_DIRS = {
    #'/home/erik/experiments/sal-classif/data/imagenet-hard-focus': 'hard.json',
    #'/home/erik/experiments/sal-classif/data/imagenet-soft-focus': 'soft.json',
    #'/home/erik/experiments/sal-classif/data/imagenet-rand-focus': 'rand.json',
    #'/home/erik/experiments/sal-classif/data/imagenet-central-focus': \
    #    'center.json',
    '/home/erik/experiments/sal-classif/data/imagenet-raw': 'raw.json',
}

# inception-v3
#MODULE_URL = 'https://tfhub.dev/google/imagenet/inception_v3/classification/1'

def get_id(path):
    return int(os.path.basename(path).split('.')[0].split('_')[-1])

def load_img(path):
    img = io.imread(path)
    if img.shape[:2] != SHAPE:
        img = transf.resize(img, SHAPE, preserve_range=True, mode='constant',
                anti_aliasing=True)
    return img

def partition(lst, size):
    parts = []
    for i in range(0, len(lst), size):
        parts.append(lst[i:i+size])
    return parts

def pre_proc(batch):
    batch = inception_v3.preprocess_input(batch)
    return batch

def _mk_batch(imgs):
    if not isinstance(imgs, (list, tuple)):
        imgs = [imgs]
    return np.stack(imgs, axis=0)

def mk_batch(paths):
    imgs = [load_img(p) for p in paths]
    batch = _mk_batch(imgs)
    batch = pre_proc(batch)
    return batch

def get_preds_dcts(preds, top=5):
    dcts = [sorted(enumerate(p), key=lambda ip: ip[1], reverse=True)[:top]\
        for p in preds]
    dcts = [{i+1: float(p) for i, p in pairs} for pairs in dcts]
    return dcts

def predict(imgs_dir, dst_path, model):
    paths = glob.glob(os.path.join(imgs_dir, '*'))
    print('on {} paths'.format(len(paths)))
    paths_parts = partition(paths, 256)

    pool = mp.Pool(4)

    #batches = pool.map(mk_batch, paths_parts)

    dct = {}
    #for paths, batch in zip(paths_parts, batches):
    #for paths in paths_parts:
    #    batch = mk_batch(paths)
    i = 1
    for paths, batch in zip(paths_parts, pool.imap(mk_batch, paths_parts)):
        print('{}/{}'.format(i, len(paths_parts)))
        i += 1
        print(batch.shape)
        preds = model.predict(batch)
        preds = get_preds_dcts(preds)
        #preds = [np.argmax(p) for p in preds]
        for p, y in zip(paths, preds):
            dct[get_id(p)] = y

    with open(dst_path, 'w') as f:
        json.dump(dct, f, indent=4, sort_keys=True)
    #print('ble')

def main():
    model = inception_v3.InceptionV3()
    for k, v in IMGS_DIRS.items():
        print(k, v)
        predict(k, v, model)

if __name__ == '__main__':
    main()
