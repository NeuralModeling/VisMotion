#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Copyright (C) 2023-2024, 类生智能, all rights reserved.
'''
----------------------------------
@Modified By:  SU Jie
@Modified:     2024-05-14 21:04:44
----------------------------------
@File:         base.py
@Version:      1.0.0
@Created:      2023-05-17 17:30:03
@Author:       SU Jie
@Description:  basic functions for training and evaluate on RDM tasks
'''


import os
import glob
import random
import yaml
import numpy as np
import torch

from torch import nn
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import MultiStepLR

from scipy.io import savemat
from tqdm.auto import tqdm

from visualize import save_plots
from dataset import RandMot, RDMData, transform
# CNN with similar structure of YuanShen
# from motnet import MotNet_Sep, MotionNet
# CNN+RNN with similar structure of YuanShen, hard to train
# from motnet import MotNet_RNN, MotionRNN
from motnet import MotNet_Conv, MotNet_Sep, MotNet_Mix, MotNet_Lin, MotNet_Mul, MotNet_RNN
from motnet import MotionNet


############################################################################
def dictcat(dest, src, keys=None):
    """concatenate dict values from src to dest.

    Args:
        dest (dict):
            the destination, will be modified.
        src (dict):
            the source dictionary.
        keys (tuple, optional): Defaults to None.
            which keys to be concatenated.
    """
    if keys is None:
        keys = src.keys()
    for key in keys:
        dest.setdefault(key, []).append(src[key])


def get_model(mpath, cfg=None, override=None, fname=None):
    """initialize or load a CNN model with given path or parameters

    Args:
        mpath (str):
            which folder to be used or which model file to be loaded.
        cfg (dict or file or None, optional): Defaults to None.
            config of the model and training process.
        override (None or bool, optional): override model config.
            True: override (if load), the new model is based on cfg;
            False: keep original model (if load from file);
            None: raise error if the model and cfg is incompatible

    Returns:
        model: MotionNet model.
        epoch: epoch number.
        saveto: folder for saving results.
    """
    cfgfile = 'config' if fname is None else fname
    mdlfile = 'model' if fname is None else fname
    if os.path.isfile(mpath):
        # this is a saved model file, we'll load it.
        saveto = os.path.dirname(mpath)
        with open(f'{saveto}/{cfgfile}.yaml', 'r') as f:
            config = yaml.safe_load(f)
        if cfg is None:
            cfg = config
        else:
            if isinstance(cfg, str):
                with open(cfg, 'r') as f:
                    cfg = yaml.safe_load(f)
            os.rename(f'{saveto}/{cfgfile}.yaml', f'{saveto}/{cfgfile}-old.yaml')
            with open(f'{saveto}/{cfgfile}.yaml', 'w') as f:
                yaml.safe_dump(cfg, f)
    else:
        if os.path.exists(mpath) and not os.path.isdir(mpath):
            raise FileExistsError(f'Invalid input {mpath}')
        saveto = mpath

        if cfg is None:
            raise ValueError('No config specified!')
        os.makedirs(saveto, exist_ok=True)
        with open(f'{saveto}/{cfgfile}.yaml', 'w') as f:
            yaml.dump(cfg, f)

    param = {}
    if 'rnn' in cfg['modeltype']:
        MotNet = MotNet_RNN
        param = {'rnn_bias': 'bias' in cfg['modeltype']}
    elif cfg['modeltype'] == 'trans':
        MotNet = MotionNet
        param = {'disturb': cfg['disturb'], 'prb': cfg['prob'], 'lvl': cfg['level']}
    elif cfg['modeltype'] == 'conv':
        MotNet = MotNet_Conv
    elif cfg['modeltype'] == 'convsep':
        MotNet = MotNet_Sep
    elif cfg['modeltype'] == 'convmix':
        MotNet = MotNet_Mix
    elif cfg['modeltype'] == 'linmix':
        MotNet = MotNet_Lin
    elif 'lin' in cfg['modeltype']:
        MotNet = MotNet_Mul
        param = {'shared': 'share' in cfg['modeltype']}

    model = MotNet(
        separate=cfg['separate'],
        average=cfg['average'],
        **param
    ).to(cfg['device'])

    if os.path.isfile(mpath):
        # this is a saved model file, we'll load it.
        epoch = model.load(mpath, not override) + 1
    else:
        with open(f'{saveto}/{mdlfile}.txt', 'w') as f:
            model.info(True, file=f)
        model.info(True)
        epoch = 1
    model.eval()

    return model, epoch, saveto, cfg


def rdm_cache(rdmpath=None, subset='3'):
    """cache the RDM dataset in memory for speed.

    reading those npz files again and again will cost losts of cpu times,
    and slow down our experiment, so here we try to cache them in memory.

    Args:
        rdmpath (str, optional): Defaults to None.
            path to the RDM dataset.
        subset (str, optional): Defaults to '3'.
            which subset to be used.

    Returns:
        list of the data dict.

    NOTE: about 20 GiB memory required for the ~1GiB dataset!!!
    """
    if rdmpath is None:
        rdmpath = f'/dev/shm/cache/rdmold'

    flist = glob.glob(f'{rdmpath}/dataset-{subset}/c*.npz')
    if len(flist) == 0:
        raise FileNotFoundError(f'No RDM data found at {rdmpath}!')
    flist.sort()
    cache = []
    trans = {
        'left':  0,
        'right': 1,
    }
    with tqdm(flist, desc='Caching', ncols=90) as pbar:
        for fname in pbar:
            data = np.load(fname)
            cache.append({
                'coherence': data['coherence'],
                'direction': trans[data['direction'].item()],
                'stim': data['stim'],
            })

    return cache


def rdm_simple(model, device='cpu', id='', cache=None, repeat=1):
    """evaluate model performance on RDM dataset.
    This function can be used to test performance of disturbed model on the whole dataset.

    Args:
        model (torch nn module):
            model for the experiment.
        device (str, optional): Defaults to 'cpu'.
            which device to be used.
        id (str, optional): Defaults to ''.
            displayed in progress bar for identification.
        cache (list or None, optional): Defaults to None.
            use the cached RDM dataset to avoid re-loading.

    Returns:
        dict: result on the RDM dataset
    """
    if cache is None:
        cache = rdm_cache()
    elif isinstance(cache, str):
        cache = rdm_cache(cache)

    model.eval()
    with torch.no_grad():
        nrdms = len(cache) * repeat

        # pre-allocate memory
        coherence = np.zeros(nrdms)
        direction = np.zeros(nrdms)
        predicted = np.zeros((nrdms, 8, 3))
        resp = np.zeros((8, nrdms))
        prob = np.zeros((8, nrdms))
        rets = np.zeros((nrdms, 6))

        for ii in tqdm(range(nrdms), desc=f'Tests {id}', ncols=90):
            data = cache[ii//repeat]
            coherence[ii] = data['coherence']
            direction[ii] = data['direction']

            # add new axis as mini batch to pass to the model
            indx = random.randrange(120-20)
            stim = data['stim'][None, None, indx:indx+20]
            frame = torch.tensor(stim).to(device).float()/255  # [0, m<=1]

            # original (white on black), reversed (black on white), [0, 1]
            # gray background with white/black dots, [0, 0.5] or [0.5, 1]
            # negative [-1, 0] and scaled version [-1, 1]
            temp = torch.cat([
                frame, 1-frame,
                0.5+frame/2, 0.5-frame/2,
                -frame, frame-1,
                2*frame - 1, 1 - 2*frame,
            ])
            pred = model(temp)

            p, r = pred.softmax(dim=1).max(dim=1)
            prob[:,ii] = p.cpu().numpy()
            resp[:,ii] = r.cpu().numpy()
            predicted[ii,:,:pred.shape[1]] = pred.cpu().numpy()

            rets[ii, 0] = coherence[ii]
            rets[ii, 1] = direction[ii] + 1
            rets[ii, 2] = resp[0, ii] + 1
            rets[ii, 3] = -np.log(prob[0, ii])  # DNN model do not have RT, so
            rets[ii, 4] = predicted[ii, 0, 0]
            rets[ii, 5] = predicted[ii, 0, 1]

        acc = [np.mean(direction == choice) for choice in resp]
        print(f'\nRDM {id:>4}  acc:', *[f'{v:.3%},' for v in acc])

    res = {
        'results':   rets,
        'coherence': coherence,
        'direction': direction,
        'predicted': predicted,
        'resp': resp,
        'prob': prob,
        'acc': acc,
    }
    return res


def train_simple(model, traindat, testsdat, cfg, epoches, saveto, trainable=None):
    if trainable is not None:
        # for transfer learning
        model.freeze('all')
        model.unfreeze(trainable)
        model.info(True)

    # optim = torch.optim.Adam(model.parameters(), lr=cfg['lrates'])
    optim = torch.optim.SGD(
        model.parameters(),
        lr=cfg['lrates'],
        momentum=cfg['momentum'],
        weight_decay=cfg['w_decay'],
        )
    sched = MultiStepLR(
        optim,
        milestones=cfg['milestone'],
        gamma=cfg['gamma'], verbose=True)
    criterion = nn.CrossEntropyLoss()

    trans = torch.tensor([0, 2, 1]).to(cfg['device'])
    # test performance before training:
    res = rdm_simple(
        model, device=cfg['device'], id='00',
        cache=testsdat, # intermediate=['fcon4', 'fcon5']
    )
    # savemat(f'{saveto}/tests-00.mat', res)
    model.save(f'{saveto}/epoch-00.pth', 0)

    for epoch in epoches:
        model.train()
        with tqdm(traindat, desc=f'Train {epoch:02d}', ncols=90) as pbar:
            for stim, param in pbar:
                if isinstance(param, tuple) or isinstance(param, list):
                    # this is the RDM dataset
                    d, c = param
                    var = (2*d-1) * c
                else:
                    # the randmot dataset
                    var = param[:,1]
                # -1, 1, 0 -> 0, 1, 2 for left, right, none
                label = trans[1+torch.sign(var).long()]

                optim.zero_grad()
                pred = model(stim[:,None])
                loss = criterion(pred, label)
                loss.backward()
                optim.step()

                with torch.no_grad():
                    resp = pred.argmax(dim=1)
                    err = (resp != label).float().mean()

                pbar.set_postfix_str(f'loss={loss:.6f}, err={err:.6f}')

        model.save(f'{saveto}/epoch-{epoch:02d}.pth', epoch)
        res = rdm_simple(model, device=cfg['device'], id=f'{epoch:02d}', cache=testsdat,)
        savemat(f'{saveto}/tests-{epoch:02d}-raw.mat', res)
        save_plots(res['results'], f'{saveto}/tests-{epoch:02d}-{res["acc"][0]:.3f}')
        sched.step()

        if epoch >= 5 and np.all(np.array(res["acc"]) == 0.5):
            print('Give up, model do not converge.')
            break


def train_dir(mdlpath, nepoch=20, noise='none', rdmpath=None):
    """train CNN model for direction classify (left, right) using random generated movie.
    """
    # config will be saved to config.yaml for reproducibility
    cfg = {
        'mdlpath':   mdlpath,
        'device':    'cuda' if torch.cuda.is_available() else 'cpu',
        'separate':  True,
        'average':   True,
        'modeltype': 'conv',     # rnn(bias), convsep, convmix, linmix, lin(share)
        'imsize':    [300, 300],
        'nframes':   20,
        # 'scaled':    [3, 7],
        # 'speed':     [5/5, 2],
        'scaled':    [2, 10],
        'speed':     [1/5, 3],
        'noise':     noise,
        'nepoch':    nepoch,
        'lrates':    0.01,
        # 'lrates':    0.005,
        'momentum':  0.9,
        'w_decay':   5e-4,
        'batchsize': 32,
        'numbatch':  500,
        'milestone': [5, 15, 30],
        # 'milestone': [10, 20, 30],
        'gamma':     0.1,
    }

    # train the model using random motion movies
    trainset = RandMot(
        shape=cfg['imsize'], scale=cfg['scaled'], speed=cfg['speed'],
        length=cfg['batchsize']*cfg['numbatch'], trans=transform(noise),
        nframes=cfg['nframes'], chans=1, device=cfg['device']
        )
    traindat = DataLoader(trainset, batch_size=cfg['batchsize'])

    model, start, saveto, _ = get_model(mdlpath, cfg, True)

    # and test on the RDM dataset
    testsdat = rdm_cache(rdmpath)
    epoches = list(range(start, start+cfg['nepoch']))
    train_simple(model, traindat, testsdat, cfg, epoches, saveto)


def train_rdm(mdlpath, nepoch=20, noise='none', rdmpath=None):
    """train CNN model for direction classify (left, right) using RDM dataset.
    NOTE: this function runs very slow! (perhaps because of loading npz files)
    """
    # config will be saved to config.yaml for reproducibility
    cfg = {
        'mdlpath':   mdlpath,
        'device':    'cuda' if torch.cuda.is_available() else 'cpu',
        'separate':  True,
        'average':   True,
        'modeltype': 'rnn',     # rnn(bias), convsep, convmix, linmix, lin(share)
        'imsize':    [300, 300],
        'split':     3,
        'thresh':    80,
        'nframes':   20,
        'invert':    False,
        'noise':     noise,
        'nepoch':    nepoch,
        'lrates':    0.01,
        'momentum':  0.9,
        'w_decay':   5e-4,
        'batchsize': 32,
        'milestone': [5, 15, 30],
        'gamma':     0.1,
    }

    # train the model using RDM dataset
    trainset = RDMData(
        root='/dev/shm/cache/rdmold' if rdmpath is None else rdmpath,
        shape=cfg['imsize'], train=True, split=cfg['split'], nframes=cfg['nframes'],
        thresh=cfg['thresh'], invert=cfg['invert'], device=cfg['device']
        )
    traindat = DataLoader(trainset, batch_size=cfg['batchsize'])

    model, start, saveto, _ = get_model(mdlpath, cfg, True)

    # and test on the RDM dataset
    testsdat = rdm_cache(rdmpath)
    epoches = list(range(start, start+cfg['nepoch']))
    train_simple(model, traindat, testsdat, cfg, epoches, saveto)


def train_vec(mdlpath, nepoch=20, noise='none'):
    """train CNN model for motion vector estimation (x,y).
    """
    # config will be saved to config.yaml for reproducibility
    cfg = {
        'mdlpath':   mdlpath,
        'device':    'cuda' if torch.cuda.is_available() else 'cpu',
        'separate':  True,
        'average':   True,
        'modeltype': 'rnn',     # rnn(bias), convsep, convmix, linmix, lin(share)
        'imsize':    [300, 300],
        'scaled':    [1, 10],
        'speed':     [0, 4],
        'noise':     noise,
        'nepoch':    nepoch,
        'lrates':    0.005,
        'momentum':  0.9,
        'w_decay':   5e-4,
        'batchsize': 64,
        'numbatch':  1000,
        'milestone': [10, 25, 40],
        'gamma':     0.1,
    }

    # train the model using random motion movies
    trainset = RandMot(
        shape=cfg['imsize'], scale=cfg['scaled'], speed=cfg['speed'],
        length=cfg['batchsize']*cfg['numbatch'], trans=transform(noise),
        nframes=cfg['nframes'], chans=1, device=cfg['device']
        )
    traindat = DataLoader(trainset, batch_size=cfg['batchsize'])

    model, start, saveto, _ = get_model(mdlpath, cfg, True)

    # and test on the RDM dataset
    testsdat = rdm_cache()

    # optim = torch.optim.Adam(model.parameters(), lr=cfg['lrates'])
    optim = torch.optim.SGD(
        model.parameters(),
        lr=cfg['lrates'],
        momentum=cfg['momentum'],
        weight_decay=cfg['w_decay'],
        )
    sched = MultiStepLR(
        optim,
        milestones=cfg['milestone'],
        gamma=cfg['gamma'], verbose=True)
    criterion = nn.MSELoss()

    for epoch in range(cfg['nepoch']):
        model.train()
        with tqdm(traindat, desc=f'Train {epoch:02d}', ncols=90) as pbar:
            for stim, speed in pbar:
                stim = stim.to(cfg['device'])
                speed = speed.to(cfg['device'])

                optim.zero_grad()
                pred = model(stim[:,None])
                loss = criterion(pred, speed)
                loss.backward()
                optim.step()

                pbar.set_postfix_str(f'loss={loss:.4f}')

        model.save(f'{saveto}/epoch-{epoch:02d}.pth', epoch)
        res = rdm_simple(model, device=cfg['device'], id=f'{epoch:02d}', cache=testsdat,)
        savemat(f'{saveto}/tests-{epoch:02d}.mat', res)
        sched.step()


def test_noisy(mdlpath, layer=None, prb=1, lvl=0, id='', rdmcache=None):
    """test on rdm dataset with disturbation of model/neurons"""
    # config will be saved to config.yaml for reproducibility

    trained, _, folder, cfg = get_model(mdlpath)
    # trained.info(True)

    cfg['modeltype'] = 'trans'
    cfg['disturb'] = layer
    cfg['prob'] = prb
    cfg['level'] = lvl
    model, _, saveto, _ = get_model(f'{folder}-{layer}', cfg, fname=id)
    # transfer from conv model for noisy experiment:
    model.from_conv(trained)
    # check the model is correctly transformed
    with torch.no_grad():
        demo = torch.randn((64,1,20,300,300), device=cfg['device'])
        res1 = trained(demo)
        res2 = model(demo)
        dif1 = (res1 - res2).abs().max().item()

        model.disturb()
        res3 = model(demo)
        dif2 = (res3 - res2).abs().max().item()
        print(dif1, dif2)

    # and test on the RDM dataset
    res = rdm_simple(model, device=cfg['device'], id=id, cache=rdmcache, repeat=2)
    savemat(f'{saveto}/{id}-raw.mat', res)
    save_plots(res['results'], f'{saveto}/{id}')
    # save_plots(res['results'], f'{saveto}/{layer}{id}-{res["acc"][0]:.3f}')
    pass


def test_layer(mdlpath, layer=None, flavor=None, cache=None, repeat=5):
    # test on the RDM dataset
    if cache is None:
        cache = rdm_cache()

    if layer is None:
        layer = [
            # neurons
            'lgn', 'v1', 'mt', 'lip',
            # connections
            'conn_l2v-all',
            'conn_v2m-all',
            'conn_m2l-all',
            'conn_l2v-weight',
            'conn_v2m-weight',
            'conn_m2l-weight',
        ]
    elif isinstance(layer, str):
        layer = [layer]

    if flavor is None:
        flavor = ['dropout', 'distavg']
    elif isinstance(flavor, str):
        flavor = [flavor]

    if 'dropout' in flavor:
        for prb in np.arange(0.00, 0.99, 0.10):
            for lyr in layer:
                for ii in range(repeat):
                    test_noisy(
                        mdlpath, layer=lyr, prb=-prb.item(), rdmcache=cache,
                        id=f'drop-{lyr}-{prb:.2f}-{ii:02d}',
                    )
    if 'distmax' in flavor:
        for lvl in np.arange(0.00, 1.01, 0.10):
            for lyr in layer:
                for ii in range(repeat):
                    test_noisy(
                        mdlpath, layer=lyr, lvl=+lvl.item(), rdmcache=cache,
                        id=f'max-{lyr}-{lvl:.1f}-{ii:02d}',
                    )
    if 'distavg' in flavor:
        # for lvl in np.arange(0.00, 1.51, 0.10):
        for lvl in np.arange(0.00, 2.01, 0.20):
            for lyr in layer:
                for ii in range(repeat):
                    test_noisy(
                        mdlpath, layer=lyr, lvl=-lvl.item(), rdmcache=cache,
                        id=f'avg-{lyr}-{lvl:.1f}-{ii:02d}',
                    )
    pass


def main():
    import sys
    print(sys.argv)

    cands = [
        'testused/conv-04/epoch-20.pth',    # tests-20-0.948-psych.svg
        'testused/conv-08/epoch-20.pth',    # tests-20-0.967-psych.svg
        'testused/conv-09/epoch-20.pth',    # tests-20-0.932-psych.svg
        'testused/conv-11/epoch-20.pth',    # tests-20-0.967-psych.svg
        'testused/conv-13/epoch-20.pth',    # tests-20-0.945-psych.svg
    ]

    if len(sys.argv) == 1:
        # for debug and test
        # torch.manual_seed(3407)
        # train_dir('testdir')
        # train_rdm('testrdm', 0, 7, 'none')
        for ii in range(50):
            train_dir(f'testused/conv-{ii:02d}')
        return

    elif len(sys.argv) == 2:
        # $0 id
        id = int(sys.argv[1])
        test_layer(cands[id])

    elif len(sys.argv) == 3:
        # $0 id layer
        id = int(sys.argv[1])
        layer = sys.argv[2]
        test_layer(cands[id], layer)
    else:
        print(f'usage: {sys.argv[0]} id [layer]')

    return


if __name__ == '__main__':
    main()

