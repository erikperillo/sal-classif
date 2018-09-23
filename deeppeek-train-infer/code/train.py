#!/usr/bin/env python3

'''
The MIT License (MIT)

Copyright (c) 2017, 2018 Erik Perillo <erik.perillo@gmail.com>

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

import numpy as np
import random
from config import train as conf

random.seed(conf['rand_seed'])
np.random.seed(conf['rand_seed'] + 1)

import tensorflow as tf
import subprocess as sp
import os
import sys
import shutil
import argparse
from collections import OrderedDict

import model
import trloop
import util

def populate_out_dir(out_dir, train_set, val_set):
    '''
    Populates output dir with info files.
    '''
    #info file
    with open(os.path.join(out_dir, 'etc', 'train-log', 'info.txt'), 'w') as f:
        print('date created (y-m-d):', util.date_str(), file=f)
        print('time created:', util.time_str(), file=f)
        print('git commit hash:', util.git_hash(), file=f)

    #saving train/val filepaths
    with open(os.path.join(out_dir, 'input', 'train.csv'), 'w') as f:
        for path in train_set:
            print(path, file=f)

    with open(os.path.join(out_dir, 'input', 'val.csv'), 'w') as f:
        for path in val_set:
            print(path, file=f)

def train():
    #parsing possible command-line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--output_dir_path', type=str, nargs='?', 
        help='path to directory to save train data',
        default=conf['output_dir_path'])
    parser.add_argument('--pre_trained_model_path', type=str, nargs='?', 
        help='path to pre-trained model',
        default=conf['pre_trained_model_path'])
    parser.add_argument('--train_set', type=str, nargs='?', 
        help='path to csv list of train set paths',
        default=conf['train_set'])
    parser.add_argument('--val_set', type=str, nargs='?', 
        help='path to csv list of validation set paths',
        default=conf['val_set'])
    args = parser.parse_args()

    #getting output_dir_path
    output_dir_path = args.output_dir_path
    #getting pre_trained_model_path
    pre_trained_model_path = args.pre_trained_model_path
    #getting train_set
    train_set = util.get_paths(args.train_set)
    #getting val
    val_set = util.get_paths(args.val_set)

    out_dir = util.mk_model_dir(output_dir_path)
    print('created out dir \'{}\', populating...'.format(out_dir),
        flush=True, end=' ')
    populate_out_dir(out_dir, train_set, val_set)
    print('done.')

    #meta-model
    meta_model_kwargs = dict(conf['meta_model_kwargs'])
    if 'rand_seed' not in meta_model_kwargs:
        meta_model_kwargs['rand_seed'] = conf['rand_seed'] + 2
    meta_model = model.MetaModel(**meta_model_kwargs)

    #creating logging object
    log = util.Tee([sys.stdout,
        open(os.path.join(out_dir, 'etc', 'train-log', 'train.log'), 'w')])

    #building graph
    if pre_trained_model_path is None:
        log.print('[info] building graph for the first time')
        graph = meta_model.build_graph()
    else:
        graph = tf.Graph()

    #tensorboard logging paths
    summ_dir = os.path.join(out_dir, 'etc', 'train-log', 'summaries')

    #training session
    with tf.Session(graph=graph) as sess:
        #if first time training, creates graph collections for model params
        #else, loads model weights and params from collections
        if pre_trained_model_path is None:
            sess.run(
                tf.group(
                    tf.global_variables_initializer(),
                    tf.local_variables_initializer()))
            meta_model.mk_params_colls(graph=graph)
        else:
            log.print('[info] loading graph/weights from \'{}\''.format(
                pre_trained_model_path))
            model.load(sess, pre_trained_model_path)
            meta_model.set_params_from_colls(graph=graph)

        #building functions
        #train function: cumputes loss
        _train_fn = meta_model.get_train_fn(sess)
        def train_fn(x, y_true):
            return _train_fn(x, y_true, {
                meta_model.params['learning_rate']: conf['learning_rate']
            })

        #test function: returns a dict with pairs metric_name: metric_value
        _test_fn = meta_model.get_test_fn(sess)
        def test_fn(x, y_true):
            metrics_values = _test_fn(x, y_true)
            return OrderedDict(zip(
                    meta_model.params['metrics'].keys(), metrics_values))

        #save model function: given epoch and iter number, saves checkpoint
        def save_model_fn(epoch=None, it=None, name=None):
            if name is None:
                path = os.path.join(out_dir, 'self', 'ckpts',
                    'epoch-{}_it-{}'.format(epoch, it))
            else:
                path = os.path.join(out_dir, 'self', 'ckpts', '{}'.format(name))
            model.save(sess, path, overwrite=True)
            print('    saved checkpoint to \'{}\''.format(path))

        #test
        if conf['use_tensorboard']:
            #tensorboard summary writers
            train_writer = tf.summary.FileWriter(
                os.path.join(summ_dir, 'train'), graph=graph)
            val_writer = tf.summary.FileWriter(
                os.path.join(summ_dir, 'val'), graph=graph)
            #running tensorboard
            cmd = ['tensorboard', '--logdir={}'.format(summ_dir)]
            cmd.extend('--{}={}'.format(k, v) \
                for k, v in conf['tensorboard_params'].items())
            log.print('[info] running \'{}\''.format(' '.join(cmd)))
            proc = sp.Popen(cmd, stdout=sp.PIPE, stderr=sp.PIPE)

            _log_fn = meta_model.get_summary_fn(sess)
            def log_fn(x, y_true, its, train=True):
                summ = _log_fn(x, y_true)
                if train:
                    train_writer.add_summary(summ, its)
                    if its%10 == 0:
                        train_writer.flush()
                else:
                    val_writer.add_summary(summ, its)
                    if its%10 == 0:
                        val_writer.flush()
        else:
            log_fn = None

        #main train loop
        print('calling train loop')
        try:
            trloop.train_loop(
                train_set=train_set,
                train_fn=train_fn,
                n_epochs=conf['n_epochs'],
                val_set=val_set,
                val_fn=test_fn,
                val_every_its=conf['val_every_its'],
                patience=conf['patience'],
                log_every_its=conf['log_every_its'],
                log_fn=log_fn,
                save_model_fn=save_model_fn,
                save_every_its=conf['save_every_its'],
                batch_gen_kw=conf['batch_gen_kw'],
                log_batch_gen_kw=conf['log_batch_gen_kw'],
                better_loss_tol=conf['better_loss_tol'],
                verbose=conf['verbose'],
                print_fn=log.print,
            )
        except KeyboardInterrupt:
            print('Keyboard Interrupt event.')
        finally:
            #closing tensorboard writers
            if conf['use_tensorboard']:
                train_writer.close()
                val_writer.close()

            #saving model on final state
            path = os.path.join(out_dir, 'self', 'ckpts', 'final')
            print('saving checkpoint to \'{}\'...'.format(path), flush=True)
            model.save(sess, path, overwrite=True)

    print('\ndone.', flush=True)

def main():
    train()

if __name__ == '__main__':
    train()
