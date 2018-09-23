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

"""
Given dataset paths, computes statistics over them.
"""

_paths = [l.strip() for l in open("../data/cat2000_train-set.csv")]
_paths = [
    "/home/erik/data/sal-dsets/cat2000/stimuli/{}.jpg".format(p) for p in _paths
]

_to_lab = not True
conf = {
    "paths": _paths,
    "n_threads": 16,
    "dst_path": "../data/cat2000_train-set_{}-stats.json".format(
        "lab" if _to_lab else "rgb"),
    "to_lab": _to_lab,
}

def sumsq(lst):
    """
    Sum of squares of list.
    """
    return sum(x**2 for x in lst)

def get_mean(n, stds, means):
    """
    Mean of means list.
    """
    return sum(means)/n

def get_std(n, stds, means):
    """
    Standard deviation from stds.
    """
    return np.sqrt(sumsq(stds)/n)# + sumsq(means)/n - (sum(means)/n)**2

def pre_proc(x):
    if x.ndim < 3:
        x = np.dstack([x, x, x])
    if conf["to_lab"]:
        x = color.rgb2lab(x)
    x = np.moveaxis(x, 2, 0)
    return x

def _get_stats(paths):
    """
    Gets stats for samples specified by paths.
    """
    n_bands = None
    for path in paths:
        bands = io.imread(path)
        bands = pre_proc(bands)
        if n_bands is None:
            # assumes x with shape (channels, ...)
            n_bands = bands.shape[0]
            stats_lst = {i: {"means": [], "stds": []} for i in range(n_bands)}
        # every sample must have the same number of bands
        assert n_bands == bands.shape[0]
        for i in range(n_bands):
            stats_lst[i]["means"].append(bands[i].mean())
            stats_lst[i]["stds"].append(bands[i].std())

    stats = {i: {
        "mean": get_mean(len(paths), s["stds"], s["means"]),
        "std": get_std(len(paths), s["stds"], s["means"]),
    } for i, s in stats_lst.items()}

    return stats

def get_stats():
    """
    Gets stats for all samples specified by paths.
    """
    if isinstance(conf["paths"], str):
        paths = util.read_csv(conf["paths"])
    else:
        paths = conf["paths"]

    #building args
    n_paths_per_thr = math.ceil(len(paths)/conf["n_threads"])
    args = []
    for i in range(conf["n_threads"]):
        args.append(paths[i*n_paths_per_thr:(i+1)*n_paths_per_thr])
    #parallelizing stats calculation
    print("working on {} filepaths...".format(len(paths)))
    pool = mp.Pool(conf["n_threads"])
    results = pool.map(_get_stats, args)

    print("merging stats..")
    stats_lst = None
    for s in results:
        if stats_lst is None:
            stats_lst = {
                i: {"means": [], "stds": []} for i in range(len(s))}
        # every stats must have same number of bands
        assert len(s) == len(stats_lst)
        for i in stats_lst.keys():
            stats_lst[i]["means"].append(s[i]["mean"])
            stats_lst[i]["stds"].append(s[i]["std"])

    assert stats_lst is not None
    stats = {i: {
        "mean": get_mean(conf["n_threads"], s["stds"], s["means"]),
        "std": get_std(conf["n_threads"], s["stds"], s["means"]),
    } for i, s in stats_lst.items()}

    print("saving to", conf["dst_path"])
    with open(conf["dst_path"], "w") as f:
        json.dump(stats, f, indent=4)

def main():
    get_stats()

if __name__ == "__main__":
    main()
