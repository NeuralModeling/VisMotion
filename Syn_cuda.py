#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Copyright (C) 2023-2024, 类生智能, all rights reserved.
'''
----------------------------------
@Modified By:  SU Jie
@Modified:     2024-03-20 17:41:09
----------------------------------
@File:         Syn_cuda.py
@Version:      1.0.0
@Created:      2023-03-18 11:32:18
@Author:       SU Jie
@Description:  synapse dynamics using pytorch with cuda support
# see as reference:
# https://compneuro.neuromatch.io/tutorials/W2D3_BiologicalNeuronModels/student/W2D3_Tutorial3.html
# https://colab.research.google.com/github/johanjan/MOOC-HPFEM-source/blob/master/LIF_ei_balance_irregularity.ipynb



import torch
import numpy as np


class Syn_Exp():
    def __init__(
            self,
            shape,
            Ve=0,           # mV
            tau=5,          # ms
            g_max=0.04,     # nS
            device='cuda',
        ):
        '''
        Expotential decay synapse model.

        presynapse, receive <= 1 spike from soma at a time,
        can be updated simutenuously with soma (neuron).

        obj.g should be collected and used
        for calculating postsynapse currents.
        '''
        if not isinstance(shape, (tuple, list)):
            shape = (shape,)
        self.Ve = Ve
        self.tau = tau
        self.g_max = g_max
        self.s = torch.zeros(shape, device=device)

    @property
    def g(self):
        return self.g_max * self.s

    def reset(self):
        self.s.fill_(0)

    def psp(self, post):
        '''
        Postsynaptic potential for calculating currents
        '''
        return post.Vm - self.Ve

    def dfunc(self):
        '''
        | d s(t)   _            s(t) |
        | ----- == > dirac() - ----- |
        |   dt     T            tau  |
        '''
        return - self.s / self.tau

    def update(self, spikes=None, dt=0.1):
        ## Forward Eular:
        # self.s.add_(self.s, alpha=-dt/self.tau)
        # or rewrite as:
        self.s.mul_(1 - dt/self.tau)

        ## 2nd order (RK2):
        # g1 = self.g - dt/2 * self.g / self.tau
        # g2 = self.g - dt * g1 / self.tau
        #    = self.g - dt * (self.g - dt/2 * self.g / self.tau) / self.tau
        #    = self.g * (1 - dt/self.tau + (dt/self.tau)**2/2)
        # RK2 (Heun) gives the same result
        # self.s *= (1 - dt/self.tau + (dt/self.tau)**2/2)
        ## or the real solution:
        # self.s *= np.exp(-dt/self.tau)
        # self.s.mul_(np.exp(-dt/self.tau))
        if spikes is not None:
            self.s.add_(spikes)


class Exp_AMPA(Syn_Exp):
    def __init__(self, shape, Ve=0, tau=3.5, g_max=0.04, device='cuda'):
        super().__init__(shape, Ve, tau, g_max, device=device)

class Exp_GABA_A(Syn_Exp):
    def __init__(self, shape, Ve=-70, tau=10, g_max=0.04, device='cuda'):
        super().__init__(shape, Ve, tau, g_max, device=device)


class Syn_NMDA():
    def __init__(
            self,
            shape,
            Ve=0,       # mV
            tau_x=2,    # ms
            tau_s=50,   # ms
            g_max=0.04, # nS
            alpha=0.5,  # kHz?
            Mg=1.2,     # mmol?
            a=0.062,    # mV^-1
            b=3.57,
            device='cuda',
        ):
        '''
        NMDA synapse with a 2-variable dynamics
        '''
        if not isinstance(shape, (tuple, list)):
            shape = (shape,)
        self.Ve = Ve
        self.tau_x = tau_x
        self.tau_s = tau_s
        self.g_max = g_max
        self.alpha = alpha
        self.Mg = Mg
        self.a = a
        self.b = b
        self.s = torch.zeros(shape, device=device)
        self.x = torch.zeros(shape, device=device)
        self._temp = torch.zeros(shape, device=device)

    @property
    def g(self):
        return self.g_max * self.s

    def reset(self):
        self.s.fill_(0)
        self.x.fill_(0)

    def psp(self, post):
        V1 = post.Vm.sub(self.Ve)
        V2 = post.Vm.mul(-self.a)
        V2.exp_().mul_(self.Mg/self.b).add_(1)
        V1.div_(V2)
        return V1
        # return (V - self.Ve) / (1 + self.Mg / self.b * torch.exp(-self.a * V))

    def dfunc(self):
        dx = - self.x / self.tau_x
        ds = - self.s / self.tau_s + self.alpha * self.x * (1 - self.s)
        return dx, ds

    def update(self, spikes=None, dt=0.1):
        # using forward Eular:

        # self.s += -dt / self.tau_s * self.s + dt * self.alpha * self.x * (1 - self.s)
        # 1. s - 1 == -(1 - s)
        torch.sub(self.s, 1, out=self._temp)
        # 2. - dt/tau_s * s
        # self.s.sub_(self.s, alpha=dt/self.tau_s)
        self.s.mul_(1 - dt/self.tau_s)
        # 3. - dt * alpha * x * (1)
        self.s.addcmul_(self.x, self._temp, value=-dt*self.alpha)

        # self.x.add_(self.x, alpha=-dt/self.tau_x)
        self.x.mul_(1 - dt/self.tau_x)
        if spikes is not None:
            self.x.add_(spikes)


def _test_synapse():
    import matplotlib.pyplot as plt
    device = 'cuda'
    ns, dt = 10000, 0.2
    syn1 = Exp_AMPA(1, device=device)
    syn2 = Exp_GABA_A(1, device=device)
    syn3 = Syn_NMDA(1, device=device)

    history = np.zeros((ns, 3))
    for ii in range(ns):
        syn1.update((ii+1)%1000 == 0, dt)
        syn2.update((ii+1)%1200 == 0, dt)
        syn3.update((ii+1)%1500 == 0, dt)
        history[ii, 0] = syn1.g.cpu().numpy()
        history[ii, 1] = syn2.g.cpu().numpy()
        history[ii, 2] = syn3.g.cpu().numpy()

    fig, ax = plt.subplots()
    ax.plot(history)
    pass


if __name__ == '__main__':
    _test_synapse()
