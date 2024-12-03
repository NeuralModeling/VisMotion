#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Copyright (C) 2023-2024, 类生智能, all rights reserved.
'''
----------------------------------
@Modified By:  SU Jie
@Modified:     2024-03-21 11:34:44
----------------------------------
@File:         LIF_cuda.py
@Version:      1.0.0
@Created:      2023-03-17 10:03:43
@Author:       SU Jie
@Description:  LIF neuron dynamics using pytorch with cuda support
'''
# see as reference:
# https://compneuro.neuromatch.io/tutorials/W2D3_BiologicalNeuronModels/student/W2D3_Tutorial1.html
# https://colab.research.google.com/github/johanjan/MOOC-HPFEM-source/blob/master/LIF_ei_balance_irregularity.ipynb


import numpy as np
import torch


################################################################################
class PoissonGroup():
    def __init__(
            self,
            shape,
            freqs,
            device='cuda',
        ) -> None:
        '''
        Neurons that randomly fire with given firing rate.
        '''
        if not isinstance(shape, (tuple, list)):
            shape = (shape,)
        self.noise = torch.zeros(shape, device=device)
        self.spike = torch.zeros(shape, dtype=bool, device=device)
        self.freqs = freqs
        self.record = []

    def update(self, dt, freqs=None, record=False):
        if freqs is None:
            freqs = self.freqs
        self.noise.uniform_()
        torch.le(self.noise, freqs * dt / 1000, out=self.spike)
        if record:
            self.record.append(self.spike.to('cpu', copy=True))
        return self.spike


################################################################################
class LIF_Basic():
    def __init__(
            self,
            shape,
            V_thresh=-50.0, # mV
            V_reset=-60.0,  # mV
            V_rest=-70.0,   # mV
            tau_ref=2.0,    # ms
            g_leak=25.0,    # nS
            Cm=500.0,       # pF; for gl in nS: tau_m=Cm/gl in ms, I in pA
            device='cuda',
        ) -> None:
        '''
        basic LIF neuron that fires when Vm cross the threshold,
        using Forward Eular to solve ODE.
        '''
        if not isinstance(shape, (tuple, list)):
            shape = (shape,)
        # self.Vm = torch.full(shape, V_init, device=device)
        self.Vm  = torch.full(shape, V_rest, device=device)
        self.Ref = torch.zeros(shape, device=device)

        self.record = []
        self.V_rest = V_rest
        self.V_reset = V_reset
        self.V_thresh = V_thresh
        self.tau_ref = tau_ref
        self.tau_m = Cm / g_leak
        self.g_leak = g_leak
        self.Cm = Cm

        self.spike = torch.zeros(shape, dtype=bool, device=device)
        self._mask = torch.zeros(shape, dtype=bool, device=device)
        self._temp = torch.zeros(shape, device=device)

    def __call__(self, Iin, dt=0.1, record=False):
        return self.update(Iin, dt=dt, record=record)

    def reset(self):
        self.Vm.fill_(self.V_rest)
        self.Ref.fill_(0)
        self.record = []

    def update(self, Iin, dt=0.1, record=False):
        # dv = ( -(V - V_L) + Iinj/g_L ) * dt / tau
        # refractory period:
        torch.lt(self.Ref, dt/2, out=self._mask)
        self.Ref.sub_(dt)

        # update LIF:
        # temp = dt/self.Cm * (Iin - self.g_leak*(self.Vm - self.V_rest))
        # 1. Vm - El
        torch.sub(self.Vm, self.V_rest, out=self._temp)
        # 2. - gl * (1)
        self._temp.mul_(-self.g_leak)
        # 3. Iin + (2)
        self._temp.add_(Iin)
        # 4. Vm += dt/Cm * (3) or 0 (refractory)
        self.Vm.addcmul_(self._temp, self._mask, value=dt/self.Cm)

        # spikes:
        torch.ge(self.Vm, self.V_thresh, out=self.spike)
        self.Vm.masked_fill_(self.spike, self.V_reset)
        self.Ref.masked_fill_(self.spike, self.tau_ref)

        if record:
            self.record.append(self.spike.to('cpu', copy=True))
        return self.spike


################################################################################
class LIF_Simple(LIF_Basic):
    def __init__(
            self,
            shape,
            V_thresh=-50.0, # mV
            V_reset=-60.0,  # mV
            V_rest=-70.0,   # mV
            tau_ref=2.0,    # ms
            g_leak=25.0,    # nS
            Cm=500.0,       # pF; for gl in nS: tau_m=Cm/gl in ms, I in pA
            device='cuda',
        ) -> None:
        '''
        basic LIF neuron with Vm show spikes when fire.
        '''
        super().__init__(shape, V_thresh, V_reset, V_rest, tau_ref, g_leak, Cm, device=device)

    def update(self, Iin, dt=0.1, record=False):
        # refractory period:
        torch.lt(self.Ref, dt/2, out=self._mask)
        self.Ref.sub_(dt)
        self.Vm.masked_fill_(~self._mask, self.V_reset)

        # update LIF:
        # temp = dt/self.Cm * (Iin - self.g_leak*(self.Vm - self.E_leak))
        # 1. Vm - El
        torch.sub(self.Vm, self.V_rest, out=self._temp)
        # 2. - gl * (1)
        self._temp.mul_(-self.g_leak)
        # 3. Iin + (2)
        self._temp.add_(Iin)
        # 4. Vm += dt/Cm * (3) or 0 (refractory)
        self.Vm.addcmul_(self._temp, self._mask, value=dt/self.Cm)

        # spikes:
        torch.ge(self.Vm, self.V_thresh, out=self.spike)
        self.Ref.masked_fill_(self.spike, self.tau_ref)
        self.Vm.add_(self.spike, alpha=20)  # such that we can see spikes

        if record:
            self.record.append(self.spike.to('cpu', copy=True))
        return self.spike


################################################################################
class LIF_GWN(LIF_Basic):
    def __init__(
            self,
            shape,
            V_thresh=-50.0, # mV
            V_reset=-60.0,  # mV
            V_rest=-70.0,   # mV
            tau_ref=2.0,    # ms
            g_leak=25.0,    # nS
            Cm=500.0,       # pF
            mu=500.0,       # pA
            sigma=10.0,
            device='cuda',
        ):
        '''
        LIF neurons with Gaussian White Noise adding to input currents
        '''
        super().__init__(shape, V_thresh, V_reset, V_rest, tau_ref, g_leak, Cm, device=device)
        self.noise = torch.zeros_like(self.Vm)
        self.mu = mu
        self.sigma = sigma

    def update(self, Iin, dt=0.1, record=False):
        # NOTE that someone said the sigma should be scaled:
        self.noise.normal_(self.mu, self.sigma / np.sqrt(dt/1000))
        # self.noise.normal_(self.mu, self.sigma)
        self.noise.add_(Iin)

        return super().update(self.noise, dt, record)


################################################################################
class LIF_OU(LIF_Basic):
    def __init__(
            self,
            shape,
            V_thresh=-50.0, # mV
            V_reset=-60.0,  # mV
            V_rest=-70.0,   # mV
            tau_ref=2.0,    # ms
            g_leak=25.0,    # nS
            Cm=500.0,       # pF
            tau_ou=10.0,    # ms
            mu=500.0,       # pA
            sigma=10.0,
            device='cuda',
        ):
        '''
        LIF neurons with Ornstein-Uhlenbeck Noise adding to input currents
        '''
        super().__init__(shape, V_thresh, V_reset, V_rest, tau_ref, g_leak, Cm, device=device)
        self.noise = torch.normal(mu, sigma, self.Vm.shape, device=device)
        self.tau_ou = tau_ou
        self.mu = mu
        self.sigma = sigma
        self._sig = np.sqrt(2/self.tau_ou) * self.sigma

    def reset(self):
        super().reset()
        self.noise.normal_(self.mu, self.sigma)

    def update(self, Iin, dt=0.1, record=False):
        # Update OU noise:
        # self.noise += dt/self.tau_ou * (self.mu - self.noise) + noise
        # 1. (OU - mu)
        torch.sub(self.noise, self.mu, out=self._temp)
        # 2. - dt/tau * (N-mu)
        self.noise.sub_(self._temp, alpha=dt/self.tau_ou)
        # 3. new gwn
        # self._temp.normal_(0, np.sqrt(dt) * np.sqrt(2/self.tau_ou) * self.sigma)
        self._temp.normal_(0, np.sqrt(dt) * self._sig)
        # 4. add to OU
        self.noise.add_(self._temp)

        # update LIF:
        # dv = ( -(self.V - self.E_L) + Iinj/self.g_L ) * dt / self.tau

        # refractory period:
        torch.lt(self.Ref, dt/2, out=self._mask)
        self.Ref.sub_(dt)

        # temp = dt/self.Cm * (Iin + self.noise - self.g_leak*(self.Vm - self.E_leak))
        # 1. Vm - El
        torch.sub(self.Vm, self.V_rest, out=self._temp)
        # 2. OU - gl * (1)
        torch.sub(self.noise, self._temp, alpha=self.g_leak, out=self._temp)
        # 3. Iin + (2)
        self._temp.add_(Iin)
        # 4. Vm += dt/Cm * (3) or 0 (refractory)
        self.Vm.addcmul_(self._temp, self._mask, value=dt/self.Cm)

        # spikes:
        torch.ge(self.Vm, self.V_thresh, out=self.spike)
        self.Vm.masked_fill_(self.spike, self.V_reset)
        self.Ref.masked_fill_(self.spike, self.tau_ref)

        if record:
            self.record.append(self.spike.to('cpu', copy=True))
        return self.spike


################################################################################
def _test_update(shape, mu=600, sigma=100, device='cuda'):
    from tqdm.auto import trange
    ns, dt = 500, 0.2
    # neuron = LIF_GWN(shape, mu=mu, sigma=sigma, device=device)
    neuron = LIF_OU(shape, mu=mu, sigma=sigma, device=device)
    for ii in trange(ns):
        neuron.update(0, dt)


def _test_line_profile(shape):
    from line_profiler import LineProfiler
    lp = LineProfiler()
    lp_wrapper = lp(_test_update)
    lp.add_function(LIF_Basic.__init__)
    lp.add_function(LIF_Basic.update)
    # lp.add_function(LIF_GWN.update)
    lp.add_function(LIF_OU.update)
    lp_wrapper(shape)
    lp.print_stats()


if __name__ == '__main__':
    # test population size and speed:
    # _test_line_profile((30000,43900))     # for 24G RTX-3090
    _test_line_profile((3000,4000))
    pass
