#!/usr/bin/env python3

import numpy as np
import os
import json
import sys
import math
import util
import glob
import multiprocessing as mp
from skimage import io
from skimage import color
import random

random.seed(10)

conf = {
    "paths": glob.glob("/home/erik/data/sal-dsets/cat2000/stimuli/*"),
    "splits": {
        "train": 0.8,
        "val": 0.2,
        #"test": 0.0,
    },
    "dst_paths": {
        "train": "../data/cat2000_train-set.csv",
        "val": "../data/cat2000_val-set.csv",
        "test": "../data/cat2000_test-set.csv",
    },
}

def get_uids(paths):
    uids = [os.path.basename(p).split(".")[0] for p in paths]
    return uids

def sample(iterable, n):
    return random.sample(iterable, min(n, len(iterable)))

def split(paths, splits, dst_paths):
    uids = set(get_uids(paths))
    dset_size = len(uids)
    for set_name, frac in splits.items():
        print("{} set:".format(set_name))
        n = round(dset_size*frac)
        uids_ = set(sample(uids, n))
        print("\tgot {} samples ({:.4f}%)".format(
            len(uids_), 100*len(uids_)/dset_size))
        uids -= uids_

        with open(dst_paths[set_name], "w") as f:
            for uid in uids_:
                print(uid, file=f)
        print("\tsaved to '{}'".format(dst_paths[set_name]))

def main():
    split(conf["paths"], conf["splits"], conf["dst_paths"])

if __name__ == "__main__":
    main()
