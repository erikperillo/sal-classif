#!/usr/bin/env python3

"""
The MIT License (MIT)

Copyright (c) 2017 Erik Perillo <erik.perillo@gmail.com>

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights to
use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies
of the Software, and to permit persons to whom the Software is furnished to do
so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS
IN THE SOFTWARE.
"""

import multiprocessing as mp
from functools import partial
import tensorflow as tf
import sys
import random
import shutil
import time
import itertools as it
import numpy as np
import os
import argparse

import util
import model
from config import infer as conf
from skimage import transform as skt

def hmirr(x):
    """
    Flips image horizontally.
    Assumes image with shape ([n_samples, n_channels,] height, width).
    """
    return x[..., ::-1]

def rot90(arr, reps=1):
    """
    Performs a 90-degree rotation (counter-clockwise) reps times.
    Assumes image with shape ([n_samples, n_channels,] height, width).
    """
    for __ in range(reps%4):
        arr = arr.swapaxes(-2, -1)[..., ::-1]
    return arr

def hmirr_averaged_predict(x, pred_fn):
    """
    Runs pred_fn for x and x reflected and calculates the average y.
    """
    _y = pred_fn(x)
    _y_hmirr = pred_fn(hmirr(x))
    y_pred = (_y + hmirr(_y_hmirr))/2
    return y_pred.astype("float32")

def rot_averaged_predict(x, pred_fn, rotations=[0]):
    """
    Runs pred_fn for each rotation of x and calculates the average y.
    """
    y_pred = np.zeros(shape=(1, ) + model.Y_SHAPE[-2:], dtype="float32")
    for rot in rotations:
        _x = rot90(x, rot)
        _y = pred_fn(_x)
        y_pred += rot90(_y, (4 - rot)%4)
    y_pred = y_pred/len(rotations)
    return y_pred.astype("float32")

def get_cut_points(length, cut_size, cut_stride, full_coverage=True):
    """
    Given a 1-d length and a cut size and stride, computes (left) points
    of the cuts.
    """
    points = list(range(0, length-cut_size+1, cut_stride))
    if full_coverage and points[-1] + cut_size < length:
        points.append(length - cut_size)
    return points

def strided_predict(x, pred_fn, shape, stride):
    """
    Runs pred_fn for each cut specified shape, striding stride.
    For positions where prediction was done more than once, calculates average.
    """
    y_pred = pred_fn(x).astype("float32")
    y_pred = y_pred.reshape((1, ) + y_pred.shape)
    return y_pred

def predict(x, model_pred_fn):
    """
    Wrapper for model predict function
    """
    x = x.reshape((1, ) + x.shape).astype("float32")
    y_pred = model_pred_fn(x)
    y_pred = y_pred.reshape(y_pred.shape[2:])
    y_pred = y_pred.clip(0, 1)
    return y_pred.astype("float32")

def get_y_pred_path(x_path, preds_dir):
    """
    Given x path and directory to save predictions, gets path of y.
    """
    filename = ".".join(os.path.basename(x_path).split(".")[:-1]) + ".png"
    path = os.path.join(preds_dir, filename)
    return path

def get_mean(lst):
    """
    Gets mean value of values of lst.
    """
    return sum(lst)/max(len(lst), 1)

def mk_preds_dir(base_dir, pattern="inference"):
    """
    Creates dir to store predictions.
    """
    #creating dir
    out_dir = util.uniq_path(base_dir, pattern)
    os.makedirs(out_dir)
    return out_dir

def load_pre_proc(path, load_fn, pre_proc_fn):
    x = load_fn(path)
    x = pre_proc_fn(x)
    return path, x

def batch_gen(paths, load_fn, pre_proc_fn):
    fn = partial(load_pre_proc, load_fn=load_fn, pre_proc_fn=pre_proc_fn)
    pool = mp.Pool(16)
    for path, x in pool.imap_unordered(fn, paths):
        yield path, x

def infer():
    """
    Main method. For paths specified in input_paths, computes prediction
    and then saves.
    input_paths can be a list of paths or a path to a directory of x files
    or a path to a csv file with paths in each line of the file.
    """
    if conf["rand_seed"] is not None:
        random.seed(conf["rand_seed"])

    #parsing possible command-line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_paths", type=str, nargs="?", 
        help="path to CSV list of input paths or input file or dir with files",
        default=conf["input_paths"])
    parser.add_argument("--output_dir_path", type=str, nargs="?", 
        help="path to directory to save predictions",
        default=conf["output_dir_path"])
    parser.add_argument("--model_path", type=str, nargs="?", 
        help="path directory containing meta-graph and weights for model",
        default=conf["model_path"])
    args = parser.parse_args()

    #getting input filepaths
    input_paths = util.get_paths(args.input_paths)
    #getting output_dir_path
    output_dir_path = args.output_dir_path
    #getting model path
    model_path = args.model_path

    if conf["max_n_preds"] is not None:
        random.shuffle(input_paths)
        input_paths = input_paths[:conf["max_n_preds"]]

    #creating base dir if needed
    if not os.path.isdir(output_dir_path):
        os.makedirs(output_dir_path)
    #creating preds dir
    preds_dir = mk_preds_dir(output_dir_path, "preds")

    #meta-model
    meta_model = model.MetaModel(**conf["meta_model_kwargs"])

    with tf.Session(graph=tf.Graph()) as sess:
        #loading model weights
        print("loading model from '{}'...".format(model_path),
            flush=True, end=" ")
        model.load(sess, model_path)
        meta_model.set_params_from_colls()
        print("done")

        #building functions
        load_fn = conf["load_fn"]
        pre_proc_fn = conf["pre_proc_fn"]
        save_y_pred_fn = conf["save_y_pred_fn"]
        #prediction function is a composition
        _pred_fn = lambda x: predict(x, meta_model.get_pred_fn(sess))
        pred_fn = (lambda x: hmirr_averaged_predict(x, _pred_fn)) \
            if conf["hmirr_averaged_predict"] else _pred_fn

        #iterating over images doing predictions
        pred_times = []
        #for path in input_paths:
        for path in input_paths:
            print("on file '{}'".format(path))

            #loading
            x = load_fn(path)
            orig_x_shape = x.shape[-2:]
            print('x shape, dtype:', x.shape, x.dtype)

            #pre-processing
            x = pre_proc_fn(x)
            print('[pre-proc] x shape, dtype:', x.shape, x.dtype)

            #predicting
            print("\tpredicting...", flush=True, end=" ")
            start_time = time.time()
            y_pred = pred_fn(x)
            pred_time = time.time() - start_time
            pred_times.append(pred_time)
            print("done. took {:.4f} seconds".format(pred_time),
                end=" | ")
            print("y_pred shape:", y_pred.shape)

            #saving
            y_pred_path = get_y_pred_path(path, preds_dir)
            save_y_pred_fn(y_pred_path, y_pred, orig_x_shape)
            print("\tsaved y_pred to '{}'".format(y_pred_path))

        print("\ndone prediction on {} files in {:.4f}s (avg {:.4f}s)".format(
            len(input_paths), sum(pred_times), get_mean(pred_times)))
        print("saved preds in '{}'".format(preds_dir))


def main():
    infer()

if __name__ == "__main__":
    main()
