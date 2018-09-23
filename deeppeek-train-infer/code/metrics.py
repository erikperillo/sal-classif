#!/usr/bin/env python3

import numpy as np
import subprocess as sp
import pandas as pd
import os.path as op
import multiprocessing as mp
import skimage.io as io
import json
import skimage.transform as skt
import glob
import sys
import os
sys.path.append(os.path.join("..", "..", "att", "deep"))
import util

def std_norm(arr):
    arr = arr.astype("float64")
    return (arr - arr.mean())/max(arr.std(), 1e-9)

def unit_norm(arr):
    arr = arr.astype("float64")
    return (arr - arr.min())/(arr.max() - arr.min())

def confusion_mtx(imap, bin_mask, divs=100, neg_bin_mask=None):
    imap = unit_norm(imap)

    positives = bin_mask.sum()
    if neg_bin_mask is None:
        neg_bin_mask = ~bin_mask
        negatives = bin_mask.shape[0]*bin_mask.shape[1] - positives
    else:
        neg_bin_mask &= ~bin_mask
        negatives = neg_bin_mask.sum()

    #print("pos={}, neg={}".format(positives, negatives))
    for thr in np.linspace(0.0, 1.0, divs):
        thr_imap = imap > thr
        tp = (thr_imap & bin_mask).sum()
        tn = (~thr_imap & neg_bin_mask).sum()
        fp = negatives - tn
        fn = positives - tp

        yield (tp, fp, tn, fn)
        #print("thr={}: tp={}, fp={}, tn={}, fn={}".format(thr, tp, fp, tn, fn))

def _auc_judd(imap, pts, divs=128):
    tpr = []
    fpr = []

    for tp, fp, tn, fn in confusion_mtx(imap, pts > 0, divs):
        tpr.append(tp/(tp + fn))
        fpr.append(fp/(fp + tn))

    auc = np.trapz(tpr[::-1], fpr[::-1]) + (1 - max(fpr))

    return auc

def auc_judd(map_filepath, pts_filepath):
    map_img, pts_img = load_and_fit_dims(map_filepath, pts_filepath)
    score = _auc_judd(map_img, pts_img)

    return score

def _auc_shuffled(imap, pts, other_pts, divs=128):
    tpr = []
    fpr = []

    for tp, fp, tn, fn in confusion_mtx(imap, pts > 0, divs, other_pts > 0):
        tpr.append(tp/(tp + fn))
        fpr.append(fp/(fp + tn))

    auc = np.trapz(tpr[::-1], fpr[::-1]) + (1 - max(fpr))

    return auc

def auc_shuffled(map_filepath, pts_filepath, other_pts_filepath):
    map_img, pts_img = load_and_fit_dims(map_filepath, pts_filepath)
    other_pts_img, __ = load_and_fit_dims(other_pts_filepath, pts_filepath)
    score = _auc_shuffled(map_img, pts_img, other_pts_img)
    return score

def _nss(sal_map, gt_pts_map):
    sal_map = std_norm(sal_map)
    gt_pts_map = gt_pts_map > 0
    return (sal_map*gt_pts_map).sum()/gt_pts_map.sum()

def nss(map_filepath, pts_filepath):
    map_img, pts_img = load_and_fit_dims(map_filepath, pts_filepath)
    score = _nss(map_img, pts_img)
    return score

def cov(a, b):
    a_mean = a.mean()
    b_mean = b.mean()
    return ((a - a_mean)*(b - b_mean)).mean()

def _cc(sal_map, gt_sal_map):
    return cov(sal_map, gt_sal_map)/(sal_map.std()*gt_sal_map.std())

def cc(map_filepath, gt_map_filepath):
    map_img, gt_map_img = load_and_fit_dims(map_filepath, gt_map_filepath)
    score = _cc(map_img, gt_map_img)

    return score

def _sim(sal_map, gt_sal_map):
    sal_map = sal_map/sal_map.sum()
    gt_sal_map = gt_sal_map/gt_sal_map.sum()
    return np.minimum(sal_map, gt_sal_map).sum()

def sim(map_filepath, gt_map_filepath):
    map_img, gt_map_img = load_and_fit_dims(map_filepath, gt_map_filepath)
    score = _sim(map_img, gt_map_img)

    return score

def _mae(sal_map, gt_sal_map):
    return abs(sal_map - gt_sal_map).mean()

def mae(map_filepath, gt_map_filepath):
    map_img, gt_map_img = load_and_fit_dims(map_filepath, gt_map_filepath)
    map_img /= map_img.max()
    gt_map_img /= gt_map_img.max()
    score = _mae(map_img, gt_map_img)

    return score

def test():
    print("in auc_judd...")
    score = auc_judd("map.jpg", "pts.jpg")
    print("done. score =", score)
    print()

    print("in auc_shuffled...")
    score = auc_shuffled("map.jpg", "pts.jpg", "other_pts.jpg")
    print("done. score =", score)
    print()
    exit()

    print("in nss...")
    score = nss("map.jpg", "pts.jpg")
    print("done. score =", score)
    print()

    print("in cc...")
    score = cc("map.jpg", "other_map.jpg")
    print("done. score =", score)
    print()

    print("in sim...")
    score = sim("map.jpg", "other_map.jpg")
    print("done. score =", score)
    print()

    print("in mae...")
    score = mae("map.jpg", "pts.jpg")
    print("done. score =", score)
    print()

#metrics to use
METRICS_FUNCS = {
    "auc_judd": _auc_judd,
    "auc_shuffled": _auc_shuffled,
    "nss": _nss,
    "mae": _mae,
    "cc": _cc,
    "sim": _sim
}

MAP_METRICS = {
    "mae",
    "cc",
    "sim",
}

PTS_METRICS = {
    "auc_judd",
    "auc_shuffled",
    "nss",
}

def compute():
    if len(sys.argv) < 2:
        print("usage: metrics.py <metric> <map> [other_args]")
        exit()

    #executing metric
    metric = sys.argv[1]
    try:
        score = METRICS_FUNCS[metric](*sys.argv[2:])
        print("%.6f" % score)
    except:
        print("_".join("(ERROR:[{}:{}])".format(sys.exc_info()[0].__name__,
            sys.exc_info()[1]).split()))

JUDD_DSET_PATH = "/home/erik/data/sal-dsets/judd"
CAT2000_DSET_PATH = "/home/erik/data/sal-dsets/cat2000"
SALICON_DSET_PATH = "/home/erik/data/sal-dsets/salicon"

METRICS = [
    "auc_judd",
    "nss",
    "mae",
    "cc",
    "sim",
]

def get_y_true_fixmap_path_cat2000(y_pred_path):
    y_pred_path = os.path.abspath(y_pred_path)
    filename = os.path.basename(y_pred_path).replace(".png", ".jpg")
    path = os.path.join(CAT2000_DSET_PATH, "maps", filename)
    return path

def get_y_true_fixpts_path_cat2000(y_pred_path):
    y_pred_path = os.path.abspath(y_pred_path)
    filename = os.path.basename(y_pred_path)
    path = os.path.join(CAT2000_DSET_PATH, "points", filename)
    return path

def get_y_true_fixmap_path_judd(y_pred_path):
    y_pred_path = os.path.abspath(y_pred_path)
    filename = os.path.basename(y_pred_path).replace(".png", "_fixMap.jpg")
    path = os.path.join(JUDD_DSET_PATH, "maps", filename)
    return path

def get_y_true_fixpts_path_judd(y_pred_path):
    y_pred_path = os.path.abspath(y_pred_path)
    filename = os.path.basename(y_pred_path).replace(".png", "_fixPts.jpg")
    path = os.path.join(JUDD_DSET_PATH, "points", filename)
    return path

def get_y_true_fixmap_path_salicon(y_pred_path):
    y_pred_path = os.path.abspath(y_pred_path)
    filename = os.path.basename(y_pred_path)
    path = os.path.join(SALICON_DSET_PATH, "maps", filename)
    return path

def get_y_true_fixpts_path_salicon(y_pred_path):
    y_pred_path = os.path.abspath(y_pred_path)
    filename = os.path.basename(y_pred_path)
    path = os.path.join(SALICON_DSET_PATH, "points", filename)
    return path

def load_fixmap(path):
    fixmap = io.imread(path)
    fixmap = unit_norm(fixmap)
    if fixmap.ndim > 2:
        fixmap = fixmap.reshape(fixmap.shape[:2])
    return fixmap

def load_fixpts(path):
    fixpts = io.imread(path)
    fixpts = unit_norm(fixpts)
    if fixpts.ndim > 2:
        fixpts = fixpts.reshape(fixpts.shape[:2])
    fixpts = (fixpts >= 0.5).astype("uint8")
    return fixpts

def get_stats(y_pred_path):
    #loading files
    y_pred = load_fixmap(y_pred_path)
    y_true_fixmap = load_fixmap(get_y_true_fixmap_path_cat2000(y_pred_path))
    y_true_fixpts = load_fixpts(get_y_true_fixpts_path_cat2000(y_pred_path))

    #reshaping
    if y_pred.shape != y_true_fixmap.shape:
        y_pred = skt.resize(
            y_pred, y_true_fixmap.shape, preserve_range=True, mode="constant")

    filename = os.path.basename(y_pred_path)
    print("[{}]".format(filename))
    dct = {}
    for metric in METRICS:
        fn = METRICS_FUNCS[metric]
        y_true = y_true_fixmap if metric in MAP_METRICS else y_true_fixpts
        val = fn(y_pred, y_true)
        dct[metric] = val
        #print("[{}] metric '{}' = {:.4f}".format(filename, metric, val))

    return dct

def main():
    paths = glob.glob("/home/erik/preds-32/*.png")
    print("working on {} paths".format(len(paths)))

    pool = mp.Pool(16)
    lines = pool.map(get_stats, paths)
    #lines = [get_stats(p) for p in paths]

    df = pd.DataFrame(lines)
    dst_path = "./results-metrics.csv"
    df.to_csv(dst_path, index=False)
    print("saved results to '{}'".format(dst_path))

    dct = {}
    for c in df.columns:
        dct[c] = df[c].mean()
    dst_path = dst_path.replace(".csv", "_means.json")
    with open(dst_path, "w") as f:
        json.dump(dct, f, indent=4, sort_keys=True)
    print("saved results means to '{}'".format(dst_path))

if __name__ == "__main__":
    main()
