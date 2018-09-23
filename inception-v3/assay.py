#!/usr/bin/env python3

import json
import os

GT_PATH = '/home/erik/data/imagenet-2012-val/ids-labels.json'
#GT_PATH = './labels.json'
YS_PATH = './preds-combs.json'

def load_json(path):
    with open(path) as f:
        dct = json.load(f)
    return dct

def load_gt():
    dct = load_json(GT_PATH)
    dct = {int(k): int(v) for k, v in dct.items()}
    return dct

def get_id(path):
    path = os.path.basename(path)
    path = path.split('.')[0].split('_')[-1].split('-')[0]
    return int(path.lstrip('0'))

def load_ys(path=YS_PATH):
    dct = load_json(path)
    dct = {get_id(k): int(v) for k, v in dct.items()}
    return dct

def main():
    gt = load_gt()
    ys = load_ys()
    keys = set(ys.keys())
    eq = [gt[k] == ys[k] for k in keys]
    print(sum(eq))
    print(sum(eq)/len(eq))

if __name__ == '__main__':
    main()
