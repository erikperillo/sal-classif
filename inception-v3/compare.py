#!/usr/bin/env python3

import json
import numpy as np
import random
from collections import OrderedDict
import os

GT_PATH = '/home/erik/data/imagenet-2012-val/ids-labels.json'
CENTRAL_PATH = './center.json'
RAND_PATH = './rand.json'
HARD_PATH = './hard.json'
SOFT_PATH = './soft.json'
RAW_PATH = './raw.json'

def load_preds(path):
    with open(path) as f:
        dct = json.load(f)
    print(len(dct))
    dct = {\
        int(k): {int(k_)-1: v_ for k_, v_ in v.items()}\
        for k, v in dct.items()
    }
    return dct

def load_gt(path, keys):
    with open(path) as f:
        dct = json.load(f)
    dct = {int(k): v for k, v in dct.items()}
    dct = {k: dct[k] for k in keys}
    return dct

def mean(l):
    return sum(l)/len(l)

def get_top_labels(prob_dct, n=1):
    items = sorted(prob_dct.items(), key=lambda lp: lp[1], reverse=True)
    labels = [i[0] for i in items][:n]
    return labels

def top_acc(y_true, y_pred, n=1):
    keys = y_pred.keys()
    correct = [y_true[k] in get_top_labels(y_pred[k], n) for k in keys]
    return mean(correct)

def top_err(y_true, y_pred, n=1):
    return 1 - top_acc(y_true, y_pred, n)

def main():
    y_central = load_preds(CENTRAL_PATH)
    y_soft = load_preds(SOFT_PATH)
    y_hard = load_preds(HARD_PATH)
    y_rand = load_preds(RAND_PATH)
    y_raw = load_preds(RAW_PATH)
    keys = set(y_central.keys())
    gt = load_gt(GT_PATH, keys)

    assert set(y_central.keys()) == set(y_soft.keys())
    assert set(y_soft.keys()) == set(y_hard.keys())
    assert set(y_hard.keys()) == set(y_rand.keys())
    assert set(y_rand.keys()) == set(y_raw.keys())
    assert set(gt.keys()) == keys

    print('n keys:', len(keys))

    #keys = list(keys)
    #random.shuffle(keys)
    #keys = keys[:1000]
    print(top_acc(gt, y_rand, 1))
    print(top_acc(gt, y_central, 1))
    print(top_acc(gt, y_hard, 1))
    print(top_acc(gt, y_soft, 1))
    print(top_acc(gt, y_raw, 1))


if __name__ == '__main__':
    main()
