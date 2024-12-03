#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Copyright (C) 2023-2024, 类生智能, all rights reserved.
'''
----------------------------------
@Modified By:  SU Jie
@Modified:     2024-03-14 11:53:31
----------------------------------
@File:         visualize.py
@Version:      1.0.0
@Created:      2023-03-15 16:17:02
@Author:       SU Jie
@Description:  plot firing and spikes for visualization


import numpy as np
import matplotlib.pyplot as plt
from scipy.io import loadmat, savemat

from logistic import psychfit, psychval


# visualize of neuron activities:
def plot_raster(spikes, dt, offset=None, ax=None):
    # spikes are np.stack(recordlist, axis=-1), t as last dimension.
    # see plot_stacked for example
    sorted = spikes.reshape(-1, spikes.shape[-1])
    trange = np.arange(spikes.shape[-1]) * dt - (0 if offset is None else offset)
    tspike = [trange[ix] for ix in sorted]

    if ax is None:
        fig, ax = plt.subplots()
    if offset is not None:
        ax.axvline(color='gray')
    ax.eventplot(tspike)
    return tspike


def plot_fire(spikes, dt, twin=50, offset=None, ax=None, **kwargs):
    # spikes are np.stack(recordlist, axis=-1), t as last dimension.
    # see plot_stacked for example
    sorted = spikes.reshape(-1, spikes.shape[-1])
    trange = np.arange(spikes.shape[-1]) * dt - (0 if offset is None else offset)
    kernel = np.ones(twin) * 1000 / (twin*dt)
    # frates = np.convolve(np.mean(sorted, axis=0), kernel, 'valid')
    frates = np.convolve(np.mean(sorted, axis=0), kernel, 'same')

    if ax is None:
        fig, ax = plt.subplots()
    if offset is not None:
        ax.axvline(color='gray')
    # ax.plot(trange[twin//2:1-twin//2], frates, **kwargs)
    ax.plot(trange, frates, **kwargs)
    return frates


def plot_stacked(record, dt, twin=50, offset=None, title=''):
    fig, ax = plt.subplots(2, sharex=True)
    ax[0].set_title(title)

    spikes = np.stack(record, axis=-1)
    plot_raster(spikes, dt, offset, ax=ax[0])

    if spikes.shape[0] == 1 or spikes.shape[0] > 3:
        plot_fire(spikes, dt, twin, offset, ax=ax[1])
        return fig

    for ii, ss in enumerate(spikes):
        plot_fire(ss, dt, twin, offset, ax=ax[1], label=f'ch-{ii}')
    ax[1].legend()
    return fig


# visualize of psychmetric curve et al.
def calc_perf(res, ax=None, norm=100):
    # stats of each cohs without considering direction
    coherence = res[:, 0]
    direction = res[:, 1]
    resp = res[:, 2]
    step = res[:, 3]
    diff = abs(res[:, 4] - res[:, 5])

    coh = np.unique(coherence)
    acc = np.full_like(coh, np.nan)
    err = np.full_like(coh, np.nan)
    art = np.full_like(coh, np.nan)
    ext = np.full_like(coh, np.nan)
    for ii, cc in enumerate(coh):
        idx = coherence == cc
        acc[ii] = np.mean(resp[idx] == (0+direction[idx]))
        err[ii] = np.mean(resp[idx] == (3-direction[idx]))
        art[ii] = np.mean(step[idx])
        ext[ii] = np.mean(diff[idx])

    if ax is not None:
        ax.plot(coh/norm, acc, 'c')
        ax.plot(coh/norm, err, 'm')
        ax.plot(coh/norm, ext, 'k')

    return coh/norm, acc, err, art, ext


def calc_psych(res, ax=None, norm=100):
    # stats of each cohs with direction seperated
    coherence = res[:, 0]
    direction = res[:, 1]
    resp = res[:, 2]
    step = res[:, 3]
    dcoh = coherence * (2*direction-3)

    cohs = np.unique(dcoh)
    birt = np.full_like(cohs, np.nan)
    rat1 = np.full_like(cohs, np.nan)
    rat2 = np.full_like(cohs, np.nan)
    for ii, cc in enumerate(cohs):
        ix = dcoh == cc
        if np.any(ix):
            birt[ii] = np.mean(step[ix])
            rat1[ii] = np.mean(resp[ix] == 1)
            rat2[ii] = np.mean(resp[ix] == 2)

    prob = (rat2 + 1 - rat1) / 2
    indx = ~np.isnan(prob)
    beta, _ = psychfit(cohs[indx]/norm, prob[indx])
    pfit = psychval(beta, cohs/norm)

    if ax is not None:
        ax.axhline(0.5, color='gray')
        ax.axvline(color='gray')
        # ax.scatter(dcoh[ix], resp[ix], 16, direction[ix], cmap='PiYG', linewidths=0, alpha=0.1)
        ax.plot(cohs/norm, prob, 'r.')
        ax.plot(cohs/norm, pfit, 'b')

    ret = {
        'dcoh': cohs/norm,
        'birt': birt,
        'prob': prob,
        'pfit': pfit,
        'k':    beta[1],
        'b':    beta[0],
    }

    return ret


def save_plots(results, fname, norm=100):
    ret = calc_psych(results, None, norm)
    coh, acc, err, art, ext = calc_perf(results, None, norm)

    fig, ax = plt.subplots(2, 2, sharex='col', sharey='row', figsize=(9,6))
    ax[0,0].axvline(color='gray')
    ax[0,0].axhline(y=0.5, color='gray', linestyle=':')
    ax[0,0].axhline(y=0, color='gray', linestyle='--')
    ax[0,0].axhline(y=1, color='gray', linestyle='--')
    ax[0,0].plot(ret['dcoh'], ret['prob'], 'r.')
    ax[0,0].plot(ret['dcoh'], ret['pfit'], 'b')
    ax[0,0].set_ylabel('percentage')
    ax[0,0].set_title(f'k={ret["k"]:.6f}, b={ret["b"]:.6f}')
    ax[1,0].axvline(color='gray')
    ax[1,0].plot(ret['dcoh'], ret['birt'], 'k.-')
    ax[1,0].set_ylabel('decision time')

    ax[0,1].axhline(y=0.5, color='gray')
    ax[0,1].plot(coh, acc, 'b.-')
    ax[0,1].plot(coh, err, 'r.-')

    ax[1,1].plot(coh, art, 'k')
    ax2 = ax[1,1].twinx()
    ax2.plot(coh, ext, 'c')

    fig.tight_layout()
    fig.savefig(f'{fname}-psych.svg')
    # ax[0,0].set_xscale('symlog')
    # ax[0,1].set_xscale('log')
    # fig.savefig(f'{fname}-psych-log.svg')

    ret.update({
        'results': results,
        'coh': coh,
        'acc': acc,
        'err': err,
        'art': art,
        'ext': ext,
    })
    savemat(f'{fname}.mat', ret)
    return ret


def _test_poisson(freqs=30, device='cuda'):
    # from LIF_torch import PoissonGroup
    from LIF_cuda import PoissonGroup

    ns, dt = 10000, 0.2
    neuron = PoissonGroup((50,50), freqs=freqs, device=device)
    for ii in range(ns):
        neuron.update(dt, record=True)

    fig = plot_stacked(
        neuron.record, dt, twin=50,
        title=f'fr={freqs} Hz'
    )
    return fig


def _test_population(Iext=100, mu=450, sigma=100, device='cuda'):
    # from LIF_cuda import LIF_GWN as LIF_Group
    from LIF_cuda import LIF_OU as LIF_Group
    # from LIF_torch import LIF_OU as LIF_Group

    ns, dt = 10000, 0.1
    neuron = LIF_Group((50,50), mu=mu, sigma=sigma, device=device)
    for ii in range(ns):
        Iin = Iext*((ii>ns//4) and (ii<ns//4*3))
        # Iin = Iext*((ii>ns//4) and (ii<ns//4*3)) + 10*(ii>ns//2)
        # Iin = Iext*((ii>ns//4) + (ii>ns//2) + (ii>ns//4*3))
        neuron.update(Iin, dt, record=True)

    fig = plot_stacked(
        neuron.record, dt, twin=50, offset=ns//4 * dt,
        title=f'I={Iext} + N({mu}, {sigma})'
    )
    return fig


if __name__ == '__main__':
    # _test_poisson(30)
    # _test_population(100, 450, 100)
    res = np.load('test_rdm.npy')
    save_plots(res, 'test_rdm')
    pass
