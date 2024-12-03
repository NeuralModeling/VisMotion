#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Copyright (C) 2023-2024, 类生智能, all rights reserved.
'''
----------------------------------
@Modified By:  SU Jie
@Modified:     2024-04-17 15:21:37
----------------------------------
@File:         simplified.py
@Version:      1.0.0
@Created:      2023-03-21 14:51:30
@Author:       SU Jie
@Description:  simplified vismodel derived from vismodel.py
'''


import time
import numpy as np

import torch
from torch.nn import functional as F
from tqdm.auto import tqdm
from tqdm.contrib import itertools

from LIF_cuda import LIF_OU as LIF_Group
from Syn_cuda import Exp_AMPA as Syn_AMPA
from Syn_cuda import Exp_GABA_A as Syn_GABA
from Syn_cuda import Syn_NMDA


############################################################################
def disturb_inplace(tensor:torch.Tensor, prb=1, lvl=0, mask=None):
    '''in-place disturb of a given tensor
    prb: how many elements to be disturbed (0<|prb|<1):
        prb<0: set to noise;
        prb>0: add noise;
    lvl: how much noise to be used:
        lvl>0: used as scaling w.r.t. max(abs(x))
        lvl<0: used as scaling w.r.t. mean(abs(x))
    '''
    # nothing should be changed if prb == 0
    if prb == 0: return None
    # keep untouched if add no noise
    if prb > 0 and lvl == 0: return None

    if mask is None:
        mask = torch.rand_like(tensor) < abs(prb)

    if prb < 0 and lvl == 0:
        # dropout: set to zero
        tensor.masked_fill_(mask, lvl)
        return mask

    if lvl > 0:
        # noise are related to max:
        sigma = lvl * tensor.abs().max()
    else:
        # lvl<0: used as sigma of gaussian (as-is), OBSOLETE.
        # noise are related to average:
        sigma = lvl * tensor.abs().mean()

    noise = torch.randn_like(tensor)
    noise.mul_(sigma)

    if prb > 0:
        # add noise to masked elements
        # tensor[mask] += noise[mask]
        tensor.addcmul_(mask, noise)
    else:
        # set to noise
        # tensor[mask] = noise[mask]
        tensor.masked_scatter_(mask, noise)
    return mask


def conn_normal(shape, avg=0.05, var=.5, prb=0, lvl=0, ratio=1, thr=0, device='cuda'):
    '''generate connection matrix with giving g_avg, std et al.,
    and optionally do disturb/perturb on the connection matrix (prob, level)
    '''
    con = torch.normal(avg, avg*var, shape, device=device)
    if ratio < 1:
        disturb_inplace(con, prb=ratio-1, lvl=0)    # probabilistic connection
    disturb_inplace(con, prb=prb, lvl=-lvl*avg)     # disturb w.r.t disired avg

    con[con<thr] = 0

    part_con = con[con != 0]
    real_avg = con.mean().item()
    real_std = con.std().item()
    part_avg = part_con.mean().item()
    part_std = part_con.std().item()
    profile = {
        # required params:
        'avg': avg,     # average
        'var': var,     # varability (w.r.t avg)
        'prb': prb,     # disturb parameter prob
        'lvl': lvl,     # disturb parameter level
        'thr': thr,     # threshold
        'ratio': ratio,
        'shape': list(shape),
        # got:
        'percent': (con!=0).float().mean().item(),  # % of non-zeros
        'partavg': part_avg,                        # effective average (non-zeros)
        'partvar': part_std/avg,                    # effective variability (non-zeros)
        'partstd': part_std,                        # effective std (non-zeros)
        'fullavg': real_avg,                        # average of full matrix
        'fullvar': real_std/avg,                    # variability w.r.t avg
        'fullstd': real_std,                        # std of full matrix
    }
    print(f'  avg={avg:.4f}, var={var:.4f}, prb={prb:.4f}, lvl={lvl:.4f}, thr={thr:.4f}, ratio={ratio:.4f}, {shape}')
    print( '  avg=%.4f, var=%.4f, pnt=%.4f, rel=%.4f, std=%.4f' % (
        profile['partavg'], profile['partvar'], profile['percent'], profile['partavg']/avg, profile['partstd'],
        ))
    print( '  avg=%.4f, var=%.4f, emp=%.4f, rel=%.4f, std=%.4f' % (
        profile['fullavg'], profile['fullvar'], 1-profile['percent'], profile['fullavg']/avg, profile['fullstd'],
        ))
    return con, profile


def spatial_kernel(
        n, width=0.3,
        alpha=1.0, beta=0.74,
        sigma_a=0.0894, sigma_b=0.1259,
    ):
    # Spatial kernel parameters from chariker2021, fig2.m
    var_a = sigma_a ** 2
    var_b = sigma_b ** 2

    xs = torch.linspace(-width, width, n)
    xx = xs**2
    dd = xx[None] + xx[:,None]

    g1 = torch.exp(-dd/var_a) / var_a / torch.pi
    g2 = torch.exp(-dd/var_b) / var_b / torch.pi
    return xs, alpha * g1 - beta * g2


def temporal_kernel(
        dt, duration=150,
        delay=0, alpha=1, beta=1,
        tau_0=3.66, tau_1=7.16,
        ):
    # temporal kernel parameters from chariker2021, fig4.m
    # add parameters a, b as scaling of two gaussian.
    tt = torch.arange(0, duration, dt)
    t1 = F.relu(tt - delay)
    t6 = t1**6
    g1 = torch.exp(-t1/tau_0) * t6 / tau_0**7
    g2 = torch.exp(-t1/tau_1) * t6 / tau_1**7
    return tt, alpha*g1 - beta*g2


############################################################################
class VisualModel():
    def __init__(self):
        '''VisualModel for motion perception (horizontal only).
        '''
        cfg = self.config()
        device = cfg['device']
        if device.startswith('cuda'):
            torch.cuda.init()

        # neurons:
        lif_E = {
            'V_thresh': cfg['V_thresh'],
            'V_reset':  cfg['V_reset'],
            'V_rest':   cfg['V_rest'],
            'Cm':       cfg['Cm_E'],
            'g_leak':   cfg['gl_E'],
            'tau_ref':  cfg['tref_E'],
            'tau_ou':   cfg['tau_ou'],
            'mu':       cfg['mu'],
            'sigma':    cfg['sigma'],
            'device':   device,
        }
        lif_I = {
            'V_thresh': cfg['V_thresh'],
            'V_reset':  cfg['V_reset'],
            'V_rest':   cfg['V_rest'],
            'Cm':       cfg['Cm_I'],
            'g_leak':   cfg['gl_I'],
            'tau_ref':  cfg['tref_I'],
            'tau_ou':   cfg['tau_ou'],
            'mu':       cfg['mu'],
            'sigma':    cfg['sigma'],
            'device':   device,
        }

        # synapse:
        ampa = {
            'Ve':       cfg['Ve_AMPA'],
            'tau':      cfg['tau_AMPA'],
            'device':   device,
        }
        gaba = {
            'Ve':       cfg['Ve_GABA'],
            'tau':      cfg['tau_GABA'],
            'device':   device,
        }
        nmda = {
            'Ve':       cfg['Ve_NMDA'],
            'tau_x':    cfg['tau_rise'],
            'tau_s':    cfg['tau_NMDA'],
            'alpha':    cfg['NMDA_alpha'],
            'Mg':       cfg['NMDA_Mg'],
            'device':   device,
        }

        ############################################################
        # spatial & temporal convolution to simulate RGC etc.
        # NOTE the result will be HUGE if kernels are not normalized
        # pre-defined conv kernels:
        size_sp = self._conv_spatial(cfg['imsize'])
        size_tp = self._conv_temporal(cfg['tsize'])
        print(f'LGN: 2x{size_sp}x{size_sp}; -> V1')
        # LGN: size_sp x size_sp. (Or should we call this RGC?)
        self.pops_lge = LIF_Group((2, size_sp, size_sp), **lif_E)
        self.ampa_lge = Syn_AMPA( (2, size_sp, size_sp), **ampa)

        # spatial convolution for calculating On/Off pairs:
        # (100x100) -> (50,50), paired on/off neurons interleaved
        size_v1 = self._matop_v1(size_sp, shape=(3,3), d=1)
        print(f'V1:  2x{size_v1[0]}x{size_v1[1]} @ (3x3,1); -> MT')
        self.pops_v1e = LIF_Group((2, *size_v1), **lif_E)
        self.ampa_v1e = Syn_AMPA( (2, *size_v1), **ampa)

        # one more spatial convolution as receptive field of MT
        # (50,50) -> (20,20) with kernel 11x11:
        # random weighted kernel:
        size_v5 = self._matop_mt(size_v1, shape=(11,11), strides=(2,2))
        print(f'MT:  2x{size_v5[0]}x{size_v5[1]} @ (11x11,2); -> LIP')
        self.pops_mte = LIF_Group((2, *size_v5), **lif_E)
        self.ampa_mte = Syn_AMPA( (2, *size_v5), **ampa)

        ############################################################
        # LIP: pooling all togethor, recurrent for decision making
        # prod(size_v5) -> num_exc, full/probabilistic connections
        scale_E = cfg['scale_E']
        scale_I = cfg['scale_I']
        num_exc = int(cfg['pop_exc'] * scale_E)
        num_inh = int(cfg['pop_inh'] * scale_I)

        # (semi-)full connection from MT to LIP excitatory:
        self.conn_m2l, self.cfg['profile_m2E'] = conn_normal(
            (2, num_exc, size_v5[0]*size_v5[1]), device=device,
            avg=cfg['avg_m2E'], var=cfg['std'], ratio=cfg['prp_m2E'],
        )

        print(f'LIP: 2x{num_exc} + {num_inh};')
        self.pops_lee = LIF_Group((2, num_exc), **lif_E)
        self.ampa_lee = Syn_AMPA( (2, num_exc), **ampa)
        self.nmda_lee = Syn_NMDA( (2, num_exc), **nmda)

        self.pops_lii = LIF_Group((1, num_inh), **lif_I)
        self.gaba_lii = Syn_GABA( (1, num_inh), **gaba)

        ############################################################
        # Hebb-strengthened weight for same/diff group
        w_pos = cfg['w_same']
        w_neg = cfg['w_diff']

        # g_avg for synaptic connections
        g_E2E = cfg['avg_E2E'] / scale_E * w_pos
        g_E2X = cfg['avg_E2E'] / scale_E * w_neg
        g_E2I = cfg['avg_E2I'] / scale_E
        g_E4E = cfg['avg_E4E'] / scale_E * w_pos
        g_E4X = cfg['avg_E4E'] / scale_E * w_neg
        g_E4I = cfg['avg_E4I'] / scale_E
        g_I2E = cfg['avg_I2E'] / scale_I
        g_I2I = cfg['avg_I2I'] / scale_I

        # ampa connections from excitatory
        # recurrent:
        print('E2E')
        self.conn_E2E, self.cfg['profile_E2E'] = conn_normal(
            (2, num_exc, num_exc), device=device,
            avg=g_E2E, var=cfg['std'],
        )
        # inter-group:
        print('E2X')
        self.conn_E2X, self.cfg['profile_E2X'] = conn_normal(
            (2, num_exc, num_exc), device=device,
            avg=g_E2X, var=cfg['std'],
        )
        # and to inhibitory
        print('E2I')
        self.conn_E2I, self.cfg['profile_E2I'] = conn_normal(
            (2, num_inh, num_exc), device=device,
            avg=g_E2I, var=cfg['std'],
            )
        # once more with nmda:
        print('E4E')
        self.conn_E4E, self.cfg['profile_E4E'] = conn_normal(
            (2, num_exc, num_exc), device=device,
            avg=g_E4E, var=cfg['std'],
        )
        print('E4X')
        self.conn_E4X, self.cfg['profile_E4X'] = conn_normal(
            (2, num_exc, num_exc), device=device,
            avg=g_E4X, var=cfg['std'],
        )
        print('E4I')
        self.conn_E4I, self.cfg['profile_E4I'] = conn_normal(
            (2, num_inh, num_exc), device=device,
            avg=g_E4I, var=cfg['std'],
        )

        # gaba connections from inhibitory:
        # to excitatory:
        print('I2E')
        self.conn_I2E, self.cfg['profile_I2E'] = conn_normal(
            (2, num_exc, num_inh), device=device,
            avg=g_I2E, var=cfg['std'],
        )
        # and recurrent:
        print('I2I')
        self.conn_I2I, self.cfg['profile_I2I'] = conn_normal(
            (1, num_inh, num_inh), device=device,
            avg=g_I2I, var=cfg['std'],
            )
        return

    def _conv_spatial(self, imsize):
        ############################################################
        # spatial convolution to simulate receptive field of LGN
        # 300x300 -> 100x100 for stride=3
        # 300x300 ->  50x50  for stride=6
        ############################################################
        # NOTE the result will be HUGE if kernels are not normalized

        cfg = self.cfg
        device = cfg['device']
        ks_width = cfg['ks_size']
        ks_steps = cfg['stride']
        ## we use a half-width padding (which is not minimum):
        # padding = ks_width // 2 - (ks_steps == 3)
        pads_sp = (0, 3, 3)
        size_sp = (imsize + 2*pads_sp[-1] - ks_width)//ks_steps + 1

        # perhaps on/off neurons should have different spatial kernel?
        _, kern_sp = spatial_kernel(
            cfg['ks_size'], width=cfg['ks_range'],
            alpha=cfg['ks_alpha'], beta=cfg['ks_beta'],
        )

        # conv3d need the tensor be [minibatch, channel, T, H, W]
        # ks = kern_sp[None,None,None]                   # [-,-,1,W,W]
        ks = kern_sp[None,None,None].to(device)

        self.convs_s = lambda x: F.conv3d(x, ks, stride=(1,ks_steps,ks_steps), padding=pads_sp)
        return size_sp

    def _conv_temporal(self, tsize=150):
        ############################################################
        # temporal convolution to simulate on/off dynamics
        # here we use full convolution to align t=0:
        # nt -> (nt + len)
        ############################################################
        cfg = self.cfg
        device = cfg['device']
        tsteps = cfg['dt'] * cfg['repeat']
        _, kern_t1 = temporal_kernel(
            tsteps, tsize, delay=cfg['on_delay'],
            alpha=cfg['on_alpha'], beta=cfg['on_beta'],
        )
        _, kern_t2 = temporal_kernel(
            tsteps, tsize, cfg['off_delay'],
            alpha=cfg['off_alpha'], beta=cfg['off_beta'],
        )

        kt = torch.concatenate([
            kern_t1.flip(0).reshape([1,1,-1,1,1]),     # [-,-,T,1,1]
            kern_t2.flip(0).reshape([1,1,-1,1,1]),     # [-,-,T,1,1]
        ]).to(device)

        self.convs_t = lambda x: F.conv3d(x, kt, padding=(kt.shape[2],0,0))
        return kt.shape[2]

    def _matop_v1(self, size_sp, shape=(3,3), d=1):
        ############################################################
        # matrix op to calculate LIP to V1 connections
        # always assuming paired on and off, so this is not flexible
        ############################################################
        cfg = self.cfg
        device = cfg['device']

        strides = (size_sp//50, size_sp//50)
        padding = (shape[0]//2, shape[1]//2)
        size_v1 = [
            (size_sp + 2*padding[0] - shape[0]) // strides[0] + 1,
            (size_sp + 2*padding[1] - shape[1]) // strides[1] + 1,
        ]
        # NOTE: fixed 1-1 connection with not variance:
        self.conn_v1, self.cfg['profile_v1'] = conn_normal(
            (4, *size_v1), device=device, var=0, avg=cfg['avg_l2v'],
        )
        def matop_v1(s_lgn:torch.Tensor):
            # interleaved pairs to make use of data
            onn1 = s_lgn[0, 0::2, 0::2]
            off1 = s_lgn[1, 0::2, d::2]
            onn2 = s_lgn[0, 1::2, d::2]
            off2 = s_lgn[1, 1::2, 0::2]

            conn_l2v = torch.stack([
                self.conn_v1[0] * onn1 + self.conn_v1[1] * off1,
                self.conn_v1[2] * onn2 + self.conn_v1[3] * off2,
            ])
            return conn_l2v
        self.matop_v1 = matop_v1

        return size_v1

    def _matop_mt(self, size_v1, shape=(11,11),strides=(2,2)):
        ############################################################
        # matrix op to calculate V1 to MT connections:
        # using unfold and matmul for conv
        ############################################################
        cfg = self.cfg
        device = cfg['device']
        padding = (0, 0)
        size_v5 = (
            (size_v1[0] + 2*padding[0] - shape[0]) // strides[0] + 1,
            (size_v1[1] + 2*padding[1] - shape[1]) // strides[1] + 1,
        )
        # random conv kernel (which makes the result unstable):
        self.conn_mt, self.cfg['profile_mt'] = conn_normal(
            (2, shape[0]*shape[1], size_v5[0]*size_v5[1]),
            device=device, var=cfg['std'], avg=cfg['avg_v2m'],
        )

        def matop_mt(s_v1:torch.Tensor):
            unfold = F.unfold(
                s_v1[:,None],
                kernel_size=shape, padding=padding, stride=strides,
            )
            unfold.mul_(self.conn_mt)
            conn_v2m = unfold.sum(dim=1).view((2,)+size_v5)
            return conn_v2m
        self.matop_mt = matop_mt

        return size_v5

    def reset(self, module=None):
        '''
        either reset and re-initialize the model for re-use,
        or re-initialize (prepare) with long enough steps.

        we can also reset part of the model in this method.
        '''
        if module is None:
            module = [
                'pops_lge',
                'ampa_lge',
                'pops_v1e',
                'ampa_v1e',
                'pops_mte',
                'ampa_mte',
                'pops_lee',
                'ampa_lee',
                'nmda_lee',
                'pops_lii',
                'gaba_lii',
            ]

        for mod in module:
            getattr(self, mod).reset()
        return

    def prepare(self, stim, steps=None, record=False):
        '''
        (re)initialize the model to make it steady.

        in this version, we also prepare visual input to the model,
        return input currents for On/Off neurons
        '''
        # print(f'Preparing model and stim...')
        cfg = self.cfg
        device = cfg['device']
        upsamp = int(cfg['ifi'] / cfg['dt'] / cfg['repeat'])

        # run with 0 input for a while to initialize the model:
        if steps is None:
            steps = cfg['steps']
        sp = []
        for _ in range(steps):
            sp.append(self.step(record=record))
        temp = np.stack(sp)
        kern = np.full(100, 10/cfg['dt'])
        fr_1 = np.convolve(temp[:,0], kern).max()
        fr_2 = np.convolve(temp[:,1], kern).max()
        if cfg['earlystop'] and max(fr_1, fr_2) >= cfg['thresh']/2:
            print(f'\nBad model, {fr_1:.4f}, {fr_2:.4f}, give up')
            return None

        # prepare stimuli via convolution:
        bright = torch.tensor(stim[None,None,:], device=device).float()
        conv_s = self.convs_s(bright / cfg['contrast'])
        # up-sample to match dt:
        conv_u = torch.repeat_interleave(conv_s, upsamp, dim=2)
        conv_t = self.convs_t(conv_u)

        current = F.relu(conv_t + cfg['background'])
        return current

    def run(self, stim, record=0):
        '''
        run one trial with visual stim
        record: 0 record nothing, faster (defaut)
                1 record spikes for plots
                <0 for a full time range simulation.
        '''
        cfg = self.cfg
        dt = cfg['dt']
        upsample = int(cfg['ifi'] / dt / cfg['repeat'])
        currents = self.prepare(stim, record=record)
        if currents is None:
            # early stop since incorrect model
            return None

        choice = '_'
        frates = np.zeros((1,2))

        for tt, rr in itertools.product(range(currents.shape[2]), range(cfg['repeat']), ncols=90, leave=False):
            if tt % upsample == 0:
                # count for each frame
                fired_l, fired_r = 0, 0

            fired = self.step(currents[0,:,tt], record=record)
            fired_l += fired[0]
            fired_r += fired[1]

            # decision threshold: one of the population A/B have FR large enough
            if rr + 1 == cfg['repeat'] and (tt+1) % upsample == 0:
                fr_l = 1000 * fired_l / self.cfg['ifi']
                fr_r = 1000 * fired_r / self.cfg['ifi']
                frates = np.append(frates, [[fr_l,fr_r]], axis=0)
                # print(tt, fr_l, fr_r)
                if tt < 5*upsample:
                    pass
                elif np.all(frates[-5:-3,0] > self.cfg['thresh']):
                    choice = 'L'
                    if record >= 0 : break
                elif np.all(frates[-5:-3,1] > self.cfg['thresh']):
                    choice = 'R'
                    if record >= 0: break

        fm_l, fm_r = np.mean(frates[-12:-2,:], axis=0)
        if choice == '_':
            # neither side reach the threshold, so let's compare which is larger
            if fm_l > fm_r:
                choice = 'l'
            else:
                choice = 'r'
        return choice, tt*cfg['repeat']*cfg['dt'], fm_l, fm_r

    def step(self, current=None, record=False):
        cfg = self.cfg
        dt = cfg['dt']
        ## 1-1 input to LGN on/off neurons
        if current is None:
            current = 0

        spike_lgn = self.pops_lge.update(current, dt, record)
        self.ampa_lge.update(spike_lgn, dt)

        ##################################################
        # calculate input current to V1 neurons:
        # each V1 neuron should connect to some paired on and off neurons.
        conn_l2v = self.matop_v1(self.ampa_lge.s)
        curr_l2v = self.ampa_lge.psp(self.pops_v1e)
        # in order to scale the differences for DS cells, we do linear transform.
        # curr_2v1 = F.relu(curr_l2v + self.cfg['add_v1e']*self.cfg['avg_l2v'])
        curr_l2v.mul_(conn_l2v).mul_(-1).add_(self.cfg['add_v1e'] * self.cfg['avg_l2v']).relu_()

        spike_v1 = self.pops_v1e.update(curr_l2v, dt, record)
        self.ampa_v1e.update(spike_v1, dt)

        ##################################################
        # calculate input current to MT neurons:
        # conv2d to make the RF of MT neurons larger with stronger differences
        conn_v2m = self.matop_mt(self.ampa_v1e.s)
        curr_v2m = self.ampa_v1e.psp(self.pops_mte)
        curr_v2m.mul_(conn_v2m).mul_(-1)

        spike_mt = self.pops_mte.update(curr_v2m, dt, record)
        self.ampa_mte.update(spike_mt, dt)

        ##################################################
        # calculate input current to LIF neurons:
        # excitation population: recieve projections from MT:
        conn_m2l = self.conn_m2l @ self.ampa_mte.s.view((2,-1,1))
        curr_m2l = self.ampa_mte.psp(self.pops_lee)
        curr_m2l.mul_(conn_m2l[:,:,0])

        # Wang's model use all-all connections, i.e., 1-matrix,
        # NOTE for torch tensor matrix multiplication:
        # (2, M, N) x (2, N) -> (2, M, 1), the 1st dim (2) be 1 by 1.
        # connections from excitatory (ampa):
        conn_e2e = self.conn_E2E @ self.ampa_lee.s[:,:,None]
        conn_e2x = self.conn_E2X @ self.ampa_lee.s[:,:,None]
        conn_e2i = self.conn_E2I @ self.ampa_lee.s[:,:,None]
        conn_x2e = conn_e2x[:,:,0].flipud()
        conn_x2e.add_(conn_e2e[:,:,0])

        # connections from excitatory (nmda):
        conn_e4e = self.conn_E4E @ self.nmda_lee.s[:,:,None]
        conn_e4x = self.conn_E4X @ self.nmda_lee.s[:,:,None]
        conn_e4i = self.conn_E4I @ self.nmda_lee.s[:,:,None]
        conn_x4e = conn_e4x[:,:,0].flipud()
        conn_x4e.add_(conn_e4e[:,:,0])

        # connections from inhibitory (gaba):
        conn_i2e = self.conn_I2E @ self.gaba_lii.s[:,:,None]
        conn_i2i = self.conn_I2I @ self.gaba_lii.s[:,:,None]

        # synaptic currents:
        curr_x2e = self.ampa_lee.psp(self.pops_lee)
        curr_x2e.mul_(conn_x2e)
        curr_x4e = self.nmda_lee.psp(self.pops_lee)
        curr_x4e.mul_(conn_x4e)
        curr_i2e = self.gaba_lii.psp(self.pops_lee)
        curr_i2e.mul_(conn_i2e[:,:,0])

        curr_e2i = self.ampa_lee.psp(self.pops_lii)
        curr_e2i.mul_(conn_e2i.sum(dim=(0,2)))
        curr_e4i = self.nmda_lee.psp(self.pops_lii)
        curr_e4i.mul_(conn_e4i.sum(dim=(0,2)))
        curr_i2i = self.gaba_lii.psp(self.pops_lii)
        curr_i2i.mul_(conn_i2i[0,:,0])

        # input current to excitatory:
        # curr_2le = self.cfg['add_exc'] - curr_m2l - curr_i2e - curr_x2e - curr_x4e
        curr_m2l.add_(curr_i2e).add_(curr_x2e).add_(curr_x4e).mul_(-1).add_(self.cfg['add_exc'])
        spike_le = self.pops_lee.update(curr_m2l, dt, record)
        self.ampa_lee.update(spike_le, dt)
        self.nmda_lee.update(spike_le, dt)

        # and to inhibitory:
        # curr_2li = self.cfg['add_inh'] - curr_i2i - curr_e2i - curr_e4i
        curr_i2i.add_(curr_e2i).add_(curr_e4i).mul_(-1).add_(self.cfg['add_inh'])
        spike_li = self.pops_lii.update(curr_i2i, dt, record)
        self.gaba_lii.update(spike_li, dt)

        return spike_le.float().mean(axis=1).cpu()

    ############################################################################
    def config(self, key=None, val=None, **kwargs):
        self.cfg = {
            'device':       'cuda',
            'steps':        500,    # 0 input steps to initialize the model
            'dt':           0.2,    # ms
            'earlystop':    False,  # stop at bad model parameters

            'imsize':       300,    # pixels
            'ifi':          16,     # ms, 60Hz; 20 for 50Hz
            'repeat':       10,     # interp frame for temporal convolution to repeat*dt = 2 ms
            'thresh':       30,     # Hz

            'contrast':     2500,   # color scale
            'background':   000,    # unused current shift

            # spatial & temporal convolution params:
            'ks_range':     0.165,
            'ks_alpha':     1.,
            'ks_beta':      1.,
            'ks_size':      9,
            'stride':       3,

            'tsize':        160,    # ms, ~80 points for 2ms resolution
            'on_delay':     10,
            'on_alpha':     1.0,
            'on_beta':      0.8,
            'off_delay':    0,
            'off_alpha':    -1.0,
            'off_beta':     -1.0,

            # LIF common params:
            'V_thresh': -50.,   # mV
            'V_reset':  -55.,   # mV
            'V_rest':   -70.,   # mV
            'tau_ou':   10.,    # ms
            'mu':       400,    # pA
            'sigma':    100,

            # Excitatory: ~OU(530, 150)
            # 'tau_E':    20,     # ms
            'Cm_E':     500.,   # pF
            'gl_E':     25.0,   # nS
            'tref_E':   2,      # ms

            # Inhibitory: ~OU(410, 70)
            # 'tau_I':    10,     # ms
            'Cm_I':     200.,   # pF
            'gl_I':     20.0,   # nS
            'tref_I':   1,      # ms

            # synapse:
            'Ve_AMPA':  0,      # mV
            'tau_AMPA': 2.,     # ms
            'Ve_GABA':  -70,    # mV
            'tau_GABA': 5.,     # ms
            'Ve_NMDA':  0,      # mV
            'tau_rise': 2.,     # ms
            'tau_NMDA': 100,    # ms
            'NMDA_Mg':  1.0,
            'NMDA_alpha':   0.5,

            ####################################################################
            # following params can be tried:
            'pop_exc':  400,
            'pop_inh':  400,
            'scale_E':  1,
            'scale_I':  1,

            # equivelent conductivity goes to the connection matrix:
            # conv connection:
            'add_v1e':  -50,    # additional input (shift) to enlarge the differences of v1 groups
            'avg_l2v':  60.0,   # 1(on) + 1(off) neurons
            # 'avg_v2m':  1.00,   # for kernel size (11,11), 121 neurons
            'avg_v2m':  2.00,   # seems to have better psychometric curve

            'add_exc':  150,    # as excitatory input to lip DS neurons
            'add_inh':  00,
            'w_same':   1.3,
            'w_diff':   0.7,

            # probabilistic connections:
            'std':      0.5,
            'prp_m2E':  0.5,
            'avg_m2E':  0.10,

            # all-all connection: num_exc synapses
            # or probabilistic connection: ~= p*num_exc
            'avg_E2E':  0.05,
            'avg_E2I':  0.04,
            'avg_E4E':  0.165,
            'avg_E4I':  0.13,
            'avg_I2E':  1.3,
            'avg_I2I':  1.0,
        }

        if key is not None:
            self.cfg[key] = val
        for k,v in kwargs.items():
            self.cfg[k] = v

        return self.cfg


################################################################################
def _test_alone(stim):
    torch.manual_seed(0)
    t0 = time.time()
    vm = VisualModel()

    for ii in range(20):
        t1 = time.time()
        vm.reset()
        p,t,v1,v2 = vm.run(stim)
        print('%s %6.1f (%7.3f,%7.3f): %6.3f' % (p,t,v1,v2, time.time()-t1))
    print(time.time()-t0)


def _test_line_profile(coh=0, dirs='r'):
    folder = '/dev/shm/cache/rdmold/dataset-3'
    temp = np.load(f'{folder}/c{abs(coh):g}{dirs}-0.npz')
    stim = temp['stim'][:60]
    if coh < 0:
        _test_alone(stim)
        return

    from line_profiler import LineProfiler
    lp = LineProfiler()
    lp_wrapper = lp(_test_alone)
    # lp.add_function(VisualModel.__init__)
    # lp.add_function(VisualModel.run)
    # lp.add_function(VisualModel.prepare)
    lp.add_function(VisualModel.step)
    # lp.add_function(LIF_Group.update)
    # lp.add_function(Syn_AMPA.psp)
    # lp.add_function(Syn_NMDA.psp)
    lp_wrapper(stim)
    lp.print_stats()


if __name__ == '__main__':
    _test_line_profile(-30, 'r')

