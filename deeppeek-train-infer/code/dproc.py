'''
The MIT License (MIT)

Copyright (c) 2017 Erik Perillo <erik.perillo@gmail.com>

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the 'Software'), to deal
in the Software without restriction, including without limitation the rights to
use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies
of the Software, and to permit persons to whom the Software is furnished to do
so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED 'AS IS', WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS
IN THE SOFTWARE.
'''

'''
Module for data processing.
every method expects images in format height, width, [channels] except
stated otherwise.
'''

from skimage import io
from skimage import transform as transf
from skimage import color
from skimage import img_as_float

import numpy as np
import math

import glob
import os

import util

def _unit_norm(x, eps=1e-6):
    return (x - x.min())/max(x.max() - x.min(), eps)

def _std_norm(x, eps=1e-6):
    return (x - x.mean())/max(x.std(), eps)

def _hwc_to_chw(img):
    if img.ndim < 3:
        return img
    return img.swapaxes(2, 1).swapaxes(1, 0)

def _chw_to_hwc(img):
    if img.ndim < 3:
        return img
    return img.swapaxes(0, 1).swapaxes(1, 2)

def _gray_to_rgb(img):
    if img.shape[0] == 3:
        return img
    return np.concatenate(3*(img, ), axis=0)

def _load(path):
    return io.imread(path)

def _get_x_path_salicon(uid, dset_path):
    path = os.path.join(dset_path, 'stimuli', '{}.jpg'.format(uid))
    return path

def _get_x_path_judd(uid, dset_path):
    path = os.path.join(dset_path, 'stimuli', '{}.jpeg'.format(uid))
    return path

def _get_x_path_cat2000(uid, dset_path):
    path = os.path.join(dset_path, 'stimuli', '{}.jpg'.format(uid))
    return path

def _get_y_path_salicon(uid, dset_path):
    path = os.path.join(dset_path, 'maps', '{}.jpg'.format(uid))
    return path

def _get_y_path_judd(uid, dset_path):
    path = os.path.join(dset_path, 'maps', '{}_fixMap.jpg'.format(uid))
    return path

def _get_y_path_cat2000(uid, dset_path):
    path = os.path.join(dset_path, 'maps', '{}.jpg'.format(uid))
    return path

def _load_x_salicon(uid, dset_path):
    path = _get_x_path_salicon(uid, dset_path)
    x = _load(path)
    if x.ndim < 3:
        x = np.dstack([x, x, x])
    x = _hwc_to_chw(x)
    return x

def _load_x_judd(uid, dset_path):
    path = _get_x_path_judd(uid, dset_path)
    x = _load(path)
    if x.ndim < 3:
        x = np.dstack([x, x, x])
    x = _hwc_to_chw(x)
    return x

def _load_x_cat2000(uid, dset_path):
    path = _get_x_path_cat2000(uid, dset_path)
    x = _load(path)
    if x.ndim < 3:
        x = np.dstack([x, x, x])
    x = _hwc_to_chw(x)
    return x

def _load_y_salicon(uid, dset_path):
    path = _get_y_path_salicon(uid, dset_path)
    y = _load(path)
    return y

def _load_y_judd(uid, dset_path):
    path = _get_y_path_judd(uid, dset_path)
    y = _load(path)
    return y

def _load_y_cat2000(uid, dset_path):
    path = _get_y_path_cat2000(uid, dset_path)
    y = _load(path)
    return y

def _load_xy_salicon(uid, dset_path):
    x = _load_x_salicon(uid, dset_path)
    y = _load_y_salicon(uid, dset_path)
    return x, y

def _load_xy_judd(uid, dset_path):
    x = _load_x_judd(uid, dset_path)
    y = _load_y_judd(uid, dset_path)
    return x, y

def _load_xy_cat2000(uid, dset_path):
    x = _load_x_cat2000(uid, dset_path)
    y = _load_y_cat2000(uid, dset_path)
    return x, y

def train_load_salicon(uid, dset_path):
    return _load_xy_salicon(uid, dset_path)

def train_load_judd(uid, dset_path):
    return _load_xy_judd(uid, dset_path)

def train_load_cat2000(uid, dset_path):
    return _load_xy_cat2000(uid, dset_path)

def infer_load(path):
    x = _load(path)
    if x.ndim < 3:
        x = np.dstack([x, x, x])
    x = _hwc_to_chw(x)
    return x

def assay_load_y_pred(path):
    pass

def assay_load_y_true(path):
    pass

def _resize_or_rescale(x, shape_or_scale):
    '''
    expects x in shape ..., height, width
    '''
    assert 2 <= x.ndim <= 3
    two_dim = x.ndim == 2
    if two_dim:
        x = x.reshape((1, ) + x.shape)

    x_dtype = x.dtype
    x_min, x_max = x.min(), x.max()

    xs = [x[i] for i in range(x.shape[0])]
    new_xs = []
    for x_ in xs:
        if isinstance(shape_or_scale, (tuple, list)):
            x_ = transf.resize(x_, shape_or_scale, preserve_range=True,
                mode='constant', order=1, anti_aliasing=True)
        else:
            x_ = transf.rescale(x_, shape_or_scale, preserve_range=True,
                mode='constant', order=1, anti_aliasing=True,
                multichannel=False)
        x_ = x_.clip(x_min, x_max).astype(x_dtype)
        new_xs.append(x_)

    new_x = np.stack(new_xs, axis=0)
    if two_dim:
        new_x = new_x[0]

    return new_x 

def resize(x, shape):
    '''
    expects x in shape ..., height, width
    '''
    return _resize_or_rescale(x, shape)

def rescale(x, scale):
    '''
    expects x in shape ..., height, width
    '''
    return _resize_or_rescale(x, scale)

def _mk_divisible(x, k):
    height, width, depth = x.shape

    pad_w = k - (width%k)
    if pad_w > 0:
        x = np.hstack([
            np.zeros(shape=(height, math.ceil(pad_w/2), depth), dtype=x.dtype),
            x,
            np.zeros(shape=(height, math.floor(pad_w/2), depth), dtype=x.dtype),
        ])
    width += pad_w

    pad_h = k - (height%k)
    if pad_h > 0:
        x = np.vstack([
            np.zeros(shape=(math.ceil(pad_h/2), width, depth), dtype=x.dtype),
            x,
            np.zeros(shape=(math.floor(pad_h/2), width, depth), dtype=x.dtype),
        ])

    return x

def _limit_shape(x, max_shape):
    height, width = x.shape[:2]
    max_height, max_width = max_shape
    scale = max(height/max_height, width/max_width)
    if scale > 1:
        x = _chw_to_hwc(rescale(_hwc_to_chw(x), 1/scale))
    return x

def align_batch_dims(batch_xy):
    '''
    expects list of xy pairs in shape ..., height, width
    '''
    shape = batch_xy[0][0].shape[-2:]
    aligned_batch_xy = []
    for x, y in batch_xy:
        if x.shape[-2:] != shape:
            x = resize(x, shape)
        if y.shape[-2:] != shape:
            y = resize(y, shape)
        aligned_batch_xy.append((x, y))
    return aligned_batch_xy

def _pre_proc_x(x, max_shape=None, to_lab=False, channel_norm=False, div=None):
    x = x.astype('float32')
    #resizing input
    if max_shape is not None:
        x = _limit_shape(x, max_shape)
    #converting to LAB colorspace
    if to_lab:
        x = color.rgb2lab(x, illuminant='D65', observer='2')
    #normalizing each channel per mean and std
    if channel_norm:
        for i in range(x.shape[0]):
            x[i] = _std_norm(x[i].astype('float32'))
    #making divisible by number
    if div is not None:
        x = _mk_divisible(x, div)
    x = x.astype('float32')
    return x

def _pre_proc_y(y, max_shape=None, div=None):
    if y.ndim < 3:
        y = y.reshape(y.shape + (1, ))
    #limiting shape
    if max_shape is not None:
        y = _limit_shape(y, max_shape)
    #unit normalization to limit y values to [0, 1]
    y = _unit_norm(y.astype('float32'))
    #making divisible by number
    if div is not None:
        y = _mk_divisible(y, div)
    y = y.astype('float32')
    return y

def _pre_proc_xy(
        xy, max_shape=None, to_lab=False, channel_norm=False, div=None):
    x, y = xy
    x = _pre_proc_x(x, max_shape, to_lab, channel_norm, div)
    y = _pre_proc_y(y, max_shape, div)
    return x, y

def train_pre_proc_xy(xy, **kwargs):
    '''
    expects input in shape channels (opt for y), height, width
    '''
    xy = _chw_to_hwc(xy[0]), _chw_to_hwc(xy[1])
    xy = _pre_proc_xy(xy, **kwargs)
    xy = _hwc_to_chw(xy[0]), _hwc_to_chw(xy[1])
    return xy

def infer_load_judd(path):
    path = '/home/erik/data/sal-dsets/judd/stimuli/{}.jpeg'.format(path)
    x = _load(path)
    if x.ndim < 3:
        x = np.dstack([x, x, x])
    x = _hwc_to_chw(x)
    return x

def infer_load_salicon(path):
    return infer_load_judd(path)

def infer_load_cat2000(path):
    return infer_load_judd(path)

def infer_pre_proc_x(x, **kwargs):
    return _hwc_to_chw(_pre_proc_x(_chw_to_hwc(x), **kwargs))

def infer_save_y_pred(path, y_pred, orig_shape=None):
    '''
    assumes y_pred comes in shape (1, h, w), in [0, 1] float dtype
    '''
    #converting to uint8 image
    y_pred = y_pred.reshape(y_pred.shape[-2:])
    if orig_shape is not None and y_pred.shape != orig_shape:
        y_pred = resize(y_pred, orig_shape)
    y_pred = y_pred.clip(0, 1)
    y_pred = (255*y_pred).astype('uint8')
    io.imsave(path, y_pred, quality=100)
