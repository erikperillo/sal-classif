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

import random
import numpy as np
import os
import glob
import json
from functools import partial

import util
import augment
import dproc
import model

def _read_stats_file(path):
    '''
    Reads JSON statistics file from path, returning a dict.
    '''
    with open(path, 'r') as f:
        dct = json.load(f)
    dct = {int(k): v for k, v in dct.items()}
    return dct

def _last_touched(dir_path, pattern='*'):
    '''
    Returns the path to file/dir in dir_path with most recent edit date.
    '''
    paths = glob.glob(os.path.join(dir_path, pattern))
    if not paths:
        return ''
    path = max(paths, key=os.path.getmtime)
    return path

### PATHS CONFIGURATION
#directory to save sessions. a session dir contains a trained model and it's
#predictions/evaluations. assumes below dir is already created
_sessions_dir_path = '/home/erik/data/ml/sessions'
if not os.path.isdir(_sessions_dir_path):
    os.makedirs(_sessions_dir_path)

#path for a to-be-created new session directory
_new_sess_dir_path = util.uniq_path(_sessions_dir_path, 'sess')
#path to the most recently modified session dir
_last_touched_sess_dir_path = _last_touched(_sessions_dir_path)

#output to save all train-related output files for a given train run
_new_train_output_dir_path = os.path.join(_new_sess_dir_path, 'trains')
_last_touched_train_output_dir_path = \
    os.path.join(_last_touched_sess_dir_path, 'trains')
#output to save all infer-related output files
_infer_output_dir_path = os.path.join(_last_touched_sess_dir_path, 'infers')

#path to base output dir of datagen pipeline
_datagen_output_dir_path = '/mnt/db/rand/croplandnet'
#path to dir that contains train/val/test set lists and train stats
dataset_dir_path = os.path.join(_datagen_output_dir_path, 'refine_20m')

#directory containing this file
_config_file_dir = os.path.dirname(os.path.abspath(__file__))
#directory containing data
_data_dir = os.path.join(os.path.dirname(_config_file_dir), 'data')

#name of dataset
_dset = 'salicon'

#path to dataset
dataset_path = '/home/erik/data/sal-dsets/{}'.format(_dset)

#path to list of train uids
_train_uids_list_path = os.path.join(
    _data_dir, '{}_train-set.csv'.format(_dset))
#path to list of val uids
_val_uids_list_path = os.path.join(
    _data_dir, '{}_val-set.csv'.format(_dset))
#path to list of test uids
_test_uids_list_path = os.path.join(
    _data_dir, '{}_test-set.csv'.format(_dset))

### DATA PROCESSING CONFIGURATION ###
#shape of input/output
_x_shape = model.X_SHAPE[-3:]
_y_shape = model.Y_SHAPE[-3:]
#maximum input shape
_train_max_shape = (288, 384)
_infer_max_shape = (336, 448)
#make inputs width/height divisible by this number
_div = 2**model.N_DOWNSAMPLE_LAYERS
#use lab colorspace
_use_lab = False
#channel-wise normalization
_channel_norm = True

#load fns
_train_load = dproc.train_load_judd if _dset == 'judd'\
    else (dproc.train_load_salicon if _dset == 'salicon'\
        else dproc.train_load_cat2000)
_infer_load = dproc.infer_load_judd if _dset == 'judd'\
    else (dproc.infer_load_salicon if _dset == 'salicon'\
        else dproc.infer_load_cat2000)

#path to statistics of train set
_stats_path = os.path.join(_data_dir,
    '{}_train-set_{}-stats.json'.format(_dset, 'lab' if _use_lab else 'rgb'))
# these stats are fed to model; they are used for pre-processing before unet
_train_set_stats = _read_stats_file(_stats_path)

#keyword arguments for meta-model
#stats is train-set statistics
# (in form of a dict {band_index: {'mean': mean, 'std': std}}
_meta_model_kwargs = {} if _channel_norm else {'stats': _train_set_stats}

#data augmentation operations
_augment_ops = [
    ('hmirr', {}, 0.5),
    ('rotation', {'angle': (-40, 40)}, 0.15),
    ('add_noise', {'max_noise_amplitude': 0.25}, 0.15),
    ('mul_noise', {'noise': (0.7, 1.3)}, 0.15),
    ('blur', {'sigma': (1.0, 2.0)}, 0.15),
    ('translation', {'transl': (-0.3, 0.3)}, 0.15),
    ('shear', {'angle': (-0.3, 0.3)}, 0.15),
    ('rescale', {'exp_scale': (-0.5, 0)}, 0.1),
]


### TRAIN CONFIGURATION ###
train = {
    #base directory where new directory with train data will be created
    #NOTE that this argument can be overriden by the command line
    'output_dir_path': _new_train_output_dir_path,

    #path to directory containing data needed by tensorflow's SavedModel,
    #can be None
    #NOTE that this argument can be overriden by the command line
    'pre_trained_model_path': \
        None,

    #list (or path to list file) with paths of train set
    #NOTE that this argument can be overriden by the command line
    'train_set': _train_uids_list_path,

    #list (or path to list file) with paths of validation set
    #NOTE that this argument can be overriden by the command line
    'val_set': _val_uids_list_path,

    #use tensorboard summaries
    'use_tensorboard': True,
    #tensorboard server parameters
    'tensorboard_params': {
        'host': '0.0.0.0',
        'port': 6006,
    },

    #model construction args
    'meta_model_kwargs': _meta_model_kwargs,

    #learning rate of the model
    'learning_rate': 1e-4,

    #number of epochs for training loop. can be None
    'n_epochs': 128,

    #logs metrics every log_every_its, can be None
    'log_every_its': 50,

    #computes metrics on validation set every val_every_its. can be None
    'val_every_its': None,

    #number of times val set loss does not improve before early stopping.
    #can be None, in which case early stopping will never occur.
    'patience': 2,

    #save checkpoint with graph/weights every save_every_its besides epochs.
    #can be None
    'save_every_its': None,

    #verbosity
    'verbose': 2,

    #arguments to be provided by trloop.batch_gen function
    'batch_gen_kw': {
        #size of batch to be fed to model
        'batch_size': 8,

        #number of fetching threads for data loading/pre-processing/augmentation
        'n_threads': 6,

        #maximum number of samples to be loaded at a time.
        #the actual number may be slightly larger due to rounding.
        'max_n_samples': 1024,

        #function to return tuple (x, y_true) given filepath
        'fetch_thr_load_fn': \
            partial(_train_load, dset_path=dataset_path),

        #function to return (possibly) augmented image
        'fetch_thr_augment_fn': \
            partial(augment.augment, operations=_augment_ops),

        #function pre-process xy
        # NOTE that there are two pre-processing steps:
        # 1. the pre-proc that happens before feeding the input to the model;
        # 2. the pre-proc that happens before feeding the input to the unet.
        # fetch_thr_pre_proc_fn refers to the first step.
        'fetch_thr_pre_proc_fn': \
            partial(dproc.train_pre_proc_xy,
                max_shape=_train_max_shape,
                to_lab=_use_lab,
                channel_norm=_channel_norm,
                div=_div,
            ),
    },

    #keyword args for log batch generator
    'log_batch_gen_kw': {
        'n_threads': 1,
        'max_n_samples': 256,
    },

    #tolerance to determine better loss
    'better_loss_tol': 1e-3,

    #random seed for reproducibility
    'rand_seed': 135582,
}


### INFER CONFIGURATION ###
infer = {
    #input filepaths.
    #it's either a list or a path (str).
    #if it's a path to a .csv file, reads it's content line by line as a list.
    #if it's a path to a directory, gets the content of the directory.
    #otherwise consider it to be a path to a single file.
    #NOTE that this argument can be overriden by the command line
    'input_paths': \
        os.path.join(dataset_dir_path, 'test_tiles.csv'),

    #base dir where new preds directory will be created
    #NOTE that this argument can be overriden by the command line
    'output_dir_path': _infer_output_dir_path,

    #path to directory containing meta-graph and weights for model
    #NOTE that this argument can be overriden by the command line
    'model_path': os.path.join(
        '/home/erik/experiments/sal-classif/data',
        'upeek-sess/trains/model-1/self/ckpts/best'
    ),

    #wether or not compute prediction for reflected image as well and
    #average the results
    'hmirr_averaged_predict': True,

    #maximum number of predictions. can be None
    'max_n_preds': None,

    #model construction args
    'meta_model_kwargs': {
    },

    #function to load input file
    'load_fn': dproc.infer_load,

    #function to load input x (before going into model)
    'pre_proc_fn': partial(dproc.infer_pre_proc_x,
        max_shape=_infer_max_shape,
        to_lab=_use_lab,
        channel_norm=_channel_norm,
        div=_div,
    ),

    #function to save prediction given path and y_pred
    'save_y_pred_fn': dproc.infer_save_y_pred,

    #random seed to be used, can be None
    'rand_seed': 88,
}
