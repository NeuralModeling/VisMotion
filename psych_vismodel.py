#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Copyright (C) 2023-2024, 类生智能, all rights reserved.
'''
----------------------------------
@Modified By:  SU Jie
@Modified:     2024-04-17 15:20:07
----------------------------------
@File:         psych_vismodel.py
@Version:      1.0.0
@Created:      2023-06-21 12:42:51
@Author:       SU Jie
@Description:  all kinds of perturbation experiments et al. on vismodel
'''


import os
import glob
import yaml
import numpy as np
import matplotlib.pyplot as plt

# from matplotlib import cm
from tqdm.auto import tqdm

from vismodel import VisualModel
from visualize import save_plots


def fcoh(fpath):
    # helper function that sort according to coherence,
    # return str like '088-l8'
    fname = os.path.basename(fpath)[:-4]
    parts = fname.split('-')
    cc = parts[0]
    id = parts[1]

    return f'{cc[1:-1]:0>3}' + '-' + cc[-1] + id


def _test_psych(key=None, val=0, repeat=2, id=3, fname=None, **kwargs):
    folder = f'/dev/shm/cache/rdmold/dataset-{id}'

    trans = {'l': 1, 'r': 2}
    flist = glob.glob(f'{folder}/c*.npz') * repeat
    # sort according to coherence such that the easiest goes first,
    # then we can quickly skip bad models
    flist.sort(key=fcoh, reverse=True)
    nfile = len(flist)

    vm = VisualModel(key=key, val=val, **kwargs)
    with open(f'{fname}.yaml', 'w') as f:
        yaml.dump(vm.cfg, f)
    ret = np.zeros((nfile, 6))
    flg = 0
    with tqdm(flist, ncols=90) as pbar:
        for ii, fn in enumerate(pbar):
            pbar.set_description_str(f'{key}={val}: {os.path.basename(fn[:-4])}')
            temp = np.load(fn)
            stim = temp['stim']

            vm.reset()
            res = vm.run(stim)
            if res is None:
                # this is a bad model that fired too much without any input
                flg = 1
                break
            choice, rt, fr1, fr2 = res
            # print(fn, choice, rt, fr1, fr2)
            # pbar.update()
            dist = abs(fr1 - fr2) / (fr1 + fr2 + 1e-6)
            if ii > 20:
                # only check for the very first stims
                pass
            elif choice == 'L' or choice == 'R':
                # decision with given threshold
                pass
            elif (fr1 + fr2) < 5:
                # model that almost static for input
                flg = 2
                if vm.cfg['earlystop']: break
            elif dist < 0.2:
                # model without a determined decision phase, no much differences
                flg = 3
                if vm.cfg['earlystop']: break

            ret[ii, 0] = temp['coherence'].item()
            ret[ii, 1] = trans[temp['direction'].item()[0]]
            ret[ii, 2] = trans[choice.lower()]
            ret[ii, 3] = rt
            ret[ii, 4] = fr1
            ret[ii, 5] = fr2

    # calculate and plot psychometric curve
    # np.save(f'{fname}.npy', ret)
    if flg != 0:
        print(f'Bad model type {flg}:', res)

    save_plots(ret, fname)
    pass


def _get_default_value(key):
    """test keys with some defaut values in one pass, which require lots of times"""
    vrange = {
        # connections:
        'w_same':   np.arange(0.20, 2.61, 0.2),     # 13
        'w_diff':   np.arange(0.20, 2.61, 0.2),     # 13

        'std':      np.arange(0.05, 1.01, 0.05),    # 20
        'prp_m2E':  np.arange(0.05, 1.01, 0.05),    # 20

        'avg_l2v':  np.arange(10.0, 151., 10),      # 15
        'avg_v2m':  np.arange(0.20, 3.01, 0.2),     # 15
        'avg_m2E':  np.arange(.020, .301, 0.020),   # 15

        'avg_E2E':  np.arange(.010, .201, 0.010),   # 20
        'avg_E4E':  np.arange(.020, .401, 0.020),   # 20
        'avg_E2I':  np.arange(.010, .201, 0.010),   # 20
        'avg_E4I':  np.arange(.020, .401, 0.020),   # 20

        # micro stimulation
        'ext_':  np.arange(5.00, 101., 5),          # 20
        'lhs_':  np.arange(5.00, 101., 5),          # 20
        'rhs_':  np.arange(5.00, 101., 5),          # 20

        # dropout & disturb:
        'prb_':  np.arange(0.05, 1.00, 0.05),       # 19
        'lvl_':  np.arange(0.10, 2.01, 0.10),       # 20
        'lev_':  np.arange(0.10, 2.01, 0.10),       # 20
    }

    if key in vrange:
        return vrange[key]
    else:
        for pat in vrange:
            if key.startswith(pat):
                return vrange[pat]
        return None


def _trans_key_vals(key, val):
    # translate key-val pairs to match model's require
    kk = key
    if key.startswith('ext_'):
        # microstim on both side
        vv = [val, val]
    elif key.startswith('lhs_'):
        kk = f'ext_{key[4:]}'
        vv = [val, 0]
    elif key.startswith('rhs_'):
        kk = f'ext_{key[4:]}'
        vv = [0, val]
    elif key.startswith('prb_'):
        # drop out of connections/neurons
        vv = -val
    elif key.startswith('lvl_'):
        # add noise to connections/neurons, w.r.t max
        vv = val
    elif key.startswith('lev_'):
        # add noise to connections/neurons, w.r.t mean
        kk = f'lvl_{key[4:]}'
        vv = -val
    else:
        vv = val
    return kk, vv


def _test_cmd_args():
    import os, sys
    print(sys.argv)

    if len(sys.argv) == 1:
        # for debug only:
        _test_psych('prb_lip', 0.1, fname=f'prb_lip-0.1')
    elif len(sys.argv) == 2:
        # $0 dataset
        rid = sys.argv[1]
        _test_psych(id=rid, fname=f'testbase/rdm{rid}')

    elif len(sys.argv) == 3:
        # $0 id key
        # call of _test_psych with key and default value from list (costs too much time)
        rid = sys.argv[1]
        key = sys.argv[2]

        vlist = _get_default_value(key)
        if vlist is None:
            print(f'invalid key={key}!')
            return

        fld = f'visval/{key}'
        os.makedirs(fld, exist_ok=True)

        print(key, len(vlist), vlist)
        for val in vlist:
            kk, vv = _trans_key_vals(key, val)
            print(f"_test_psych({kk}, {vv}, fname='{fld}/{key}-{val:.3f}-{rid}')")
            _test_psych(kk, vv, fname=f'{fld}/{key}-{val:.3f}-{rid}')
            plt.close('all')

    elif len(sys.argv) == 4:
        # $0 id key val
        # direct call of _test_psych with key, val specified from command line
        rid = sys.argv[1]
        key = sys.argv[2]
        val = sys.argv[3]

        fld = f'visval/{key}'
        os.makedirs(fld, exist_ok=True)

        kk, vv = _trans_key_vals(key, float(val))
        print(f"_test_psych({kk}, {vv}, fname='{fld}/{key}-{val}-{rid}')")
        _test_psych(kk, vv, fname=f'{fld}/{key}-{val}-{rid}')

    elif len(sys.argv) == 6:
        # $0 id key1 val1 key2 val2
        # call with two key, val pairs specified from command line
        rid = sys.argv[1]
        key1 = sys.argv[2]
        val1 = sys.argv[3]
        key2 = sys.argv[4]
        val2 = sys.argv[5]

        fld = f'vispair/{key1}-{key2}'
        os.makedirs(fld, exist_ok=True)

        k1, v1 = _trans_key_vals(key1, float(val1))
        k2, v2 = _trans_key_vals(key2, float(val2))
        params = {k1: v1, k2: v2}
        # print(params)
        print(f"_test_psych(fname='{fld}/{key1}-{val1}_{key2}-{val2}-{rid}', {params})")
        _test_psych(fname=f'{fld}/{key1}-{val1}_{key2}-{val2}-{rid}', **params)

    else:
        print(f'usage: {sys.argv[0]} id key val [key] [val]')
    pass


if __name__ == '__main__':
    _test_cmd_args()
