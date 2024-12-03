#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Copyright (C) 2023-2024, 类生智能, all rights reserved.
'''
----------------------------------
@Modified By:  SU Jie
@Modified:     2024-05-14 12:06:30
----------------------------------
@File:         motnet.py
@Version:      2.0.0
@Created:      2023-05-23 17:29:01
@Author:       SU Jie
@Description:  some CNN models and noisy data argumentation for training.
'''


import math
import torch
from torch import nn

from vismodel import disturb_inplace

############################################################################
class BaseModel(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, *args):
        raise NotImplementedError('Not implement in subclass!')

    def freeze(self, layers, freeze=True):
        for layer in self._modules:
            if ((isinstance(layers, list) and layer in layers) or
                (isinstance(layers, str) and (layers in layer or layers=='all'))
                ):
                module = getattr(self, layer)
                for param in module.parameters():
                    param.requires_grad = not freeze

    def unfreeze(self, layers):
        self.freeze(layers, False)

    def info(self, layers=False, file=None):
        """show the details of the model, number of parameters in each layer

        Args:
            layers (bool, optional): Defaults to False.
                display the model (each layer).

        Returns:
            int: number of adjustable parameters of the model.
        """
        if layers:
            print(self, file=file)
        counts = 0
        for name, parameter in self.named_parameters():
            params = parameter.numel()
            if not parameter.requires_grad:
                print(f' {name:<20} {params:+}\t{list(parameter.shape)}', file=file)
            else:
                print(f' {name:<21} {params}\t{list(parameter.shape)}', file=file)
                counts += params

        print('-'*21, file=file)
        print(f"Trainable Parameters:  {counts}", file=file)
        return counts

    def save(self, fname, epoch):
        """save the model to file

        Args:
            fname (str): file name to be saved to.
            epoch (int): the epoch number.
        """
        torch.save({
            'epoch':  epoch,
            'states': self.state_dict(),
        }, fname)

    def load(self, fname, strict=True):
        """load model from file and return the number of epoches

        Args:
            fname (str): which file to be loaded.
            strict (bool, optional): Defaults to True.
                require strict match or ignore incompatible layers.

        Returns:
            int: the epoch number at saving.
        """
        if isinstance(fname, str):
            print(f'Loading from {fname}...')
            loaded = torch.load(fname)
            states = loaded['states']
            epoch = loaded['epoch']
        else:
            # assuming fname is the well loaded state dict
            states = fname
            epoch = None

        selfstate = self.state_dict()
        for name, param in states.items():
            loaded = False
            if name in selfstate:
                s1 = selfstate[name].shape
                s2 = param.shape
                if s1 == s2:
                    selfstate[name].copy_(param)
                    loaded = True
                elif strict == False and selfstate[name].ndim == param.ndim:
                    print(f'partial load unmatched param {name}')
                    ss = [s1[i] if s1[i]<=s2[i] else s2[i] for i in range(param.ndim)]
                    if param.ndim == 1:
                        selfstate[name][:ss[0]] = param[:ss[0]]
                    if param.ndim == 2:
                        selfstate[name][:ss[0],:ss[1]] = param[:ss[0],:ss[1]]
                    elif param.ndim == 3:
                        selfstate[name][:ss[0],:ss[1],:ss[2]] = param[:ss[0],:ss[1],:ss[2]]
                    elif param.ndim == 4:
                        selfstate[name][:ss[0],:ss[1],:ss[2],:ss[3]] = param[:ss[0],:ss[1],:ss[2],:ss[3]]
                    else:
                        raise NotImplementedError('Unsupported tensor size!')
                    loaded = True
            if not loaded:
                if strict:
                    raise ValueError(f'incompatible model! No {name} or size unmatched!')
                else:
                    print(f'ignore unmatched param {name}')
        return epoch

class DotLinear(nn.Module):
    def __init__(self, shape, reduce=None, bias=True):
        super().__init__()
        self.shape = shape
        self.reduce = reduce
        self.weight = nn.Parameter(torch.Tensor(*shape))
        if bias:
            size = list(shape)
            if reduce is not None:
                size[reduce] = 1
            self.bias = nn.Parameter(torch.Tensor(*size))
        else:
            self.register_parameter("bias", None)

        self._reset()

    def extra_repr(self) -> str:
        s = f'shape={self.shape}'
        s += f', reduce={self.reduce}' if self.reduce is not None else ''
        s += ', bias=True' if self.bias is not None else ''
        return s

    def _reset(self):
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in)
            nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, input):
        output = input * self.weight
        if self.reduce is not None:
            output = output.sum(dim=self.reduce, keepdim=True)
        if self.bias is not None:
            output += self.bias
        return output


class MulLinear(nn.Module):
    def __init__(self, shape, bias=True):
        super().__init__()
        self.shape = shape
        self.weight = nn.Parameter(torch.Tensor(*shape))
        if bias:
            size = list(shape)
            size[-2] = 1
            self.bias = nn.Parameter(torch.Tensor(*size))
        else:
            self.register_parameter("bias", None)

        self._reset()

    def extra_repr(self) -> str:
        s = f'shape={self.shape}'
        s += ', bias=True' if self.bias is not None else ''
        return s

    def _reset(self):
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in)
            nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, input):
        output = torch.matmul(input, self.weight)
        if self.bias is not None:
            output += self.bias
        return output



class MotNet_Simple(BaseModel):
    def __init__(self, ks=6, ss=3, kt=10, lipsize=400):
        super().__init__()
        # firstly, the same conv_s and conv_t before LGN
        # input should be N x 1 x T x 300x300
        #              batch chan T    H   W
        self.conv_s = nn.Conv3d(1, 1, kernel_size=( 1,ks,ks), padding=(0,3,3), stride=(1,ss,ss), bias=False)
        self.conv_t = nn.Conv3d(1, 2, kernel_size=(kt, 1, 1), padding=(0,0,0), stride=(1,1,1))
        self.fire_lgn = nn.ReLU()

        self.conv_v1 = nn.Conv3d(2, 2, kernel_size=(1,3,3), padding=(0,1,1), stride=(1,6//ss,6//ss))
        self.fire_v1 = nn.ReLU()

        self.conv_mt = nn.Conv3d(2, 2, kernel_size=(1,11,11), padding=(0,0,0), stride=(1,2,2), groups=2)
        self.fire_mt = nn.ReLU()

        self.conv_m2l = nn.Conv3d(2, 2*lipsize, kernel_size=(1,20,20), groups=2)
        self.pool_lip = nn.AdaptiveAvgPool3d((1, None, None))
        self.fire_lip = nn.ReLU()

        # finally the output layer
        self.decision = nn.AdaptiveAvgPool2d((2,1))
        # self.decision = nn.Conv1d(2*lipsize, 2, kernel_size=1, groups=2, bias=False)

    def forward(self, stim):
        # visual stim -> spatial & temporal convolution:
        conv_s = self.conv_s(stim)
        # fire_lgn = self.fire_lgn(conv_s)
        conv_t = self.conv_t(conv_s)
        fire_lgn = self.fire_lgn(conv_t)

        conv_v1 = self.conv_v1(fire_lgn)
        fire_v1 = self.fire_v1(conv_v1)

        conv_mt = self.conv_mt(fire_v1)
        fire_mt = self.fire_mt(conv_mt)

        temp_lip = self.conv_m2l(fire_mt)
        curr_lip = self.pool_lip(temp_lip)
        fire_lip = self.fire_lip(curr_lip)

        # collect for final decision
        predicts = self.decision(fire_lip.flatten(2)).squeeze()
        return predicts


class MotNet_Conv(BaseModel):
    '''
    Basic CNN model with the same structure of vismodel,
    separate connections from MT DS groups to LIP DS groups,
    see vismodel.py, rdmfull_simplified.py for comparison
    see also: MotNet_Sep, MotNet_Lin, MotionNet.
    '''
    def __init__(
            self, separate=True, average=True, pooling='lip',
            ks=9, ss=3, kt=10, lipsize=400,
        ):
        super().__init__()
        self.separate = separate
        self.average = average
        self.pooling = pooling

        # firstly, the same conv_s and conv_t before LGN
        # input should be N x 1 x T x 300x300
        #              batch chan T    H   W
        if separate:
            self.conv_s = nn.Conv3d(1, 1, kernel_size=( 1,ks,ks), padding=(0,3,3), stride=(1,ss,ss), bias=False)
            # became N x 1 x T x 100 x 100 (or 50x50 for stride=6)
            self.conv_t = nn.Conv3d(1, 2, kernel_size=(kt, 1, 1), padding=(0,0,0), stride=(1,1,1))
            # became N x 2 x t x 100 x 100 (or 50x50)
        else:
            self.conv_0 = nn.Conv3d(1, 2, kernel_size=(kt,ks,ks), padding=(0,3,3), stride=(1,ss,ss))

        # then activity of LGN:
        self.fire_lgn = nn.ReLU()

        # then V1 and its activity (N x 2 x t x 50 x 50)
        self.conv_v1 = nn.Conv3d(2, 2, kernel_size=(1,3,3), padding=(0,1,1), stride=(1,6//ss,6//ss))
        self.fire_v1 = nn.ReLU()

        # then MT and its activity (N x 2 x t x 20 x 20)
        self.conv_mt = nn.Conv3d(2, 2, kernel_size=(1,11,11), padding=(0,0,0), stride=(1,2,2), groups=2)
        if pooling == 'mt':
            self.pool_mt = nn.AdaptiveAvgPool3d((1, None, None))
        self.fire_mt = nn.ReLU()

        # then LIP and activity (N x 2 x t x 20 x 20) -> (N x 2x400 x t)
        # using built-in modules for grouped linear is not intuitive:
        # groups=2, such that each channel gives 400 results:
        self.conv_m2l = nn.Conv3d(2, 2*lipsize, kernel_size=(1,20,20), groups=2)
        if pooling == 'lip':
            self.pool_lip = nn.AdaptiveAvgPool3d((1, None, None))
        self.fire_lip = nn.ReLU()

        # finally the output layer
        if average:
            self.decision = nn.AdaptiveAvgPool2d((2,1))
        else:
            # groups=2, such that 800 -> 2 be (400+400) -> (1+1)
            self.decision = nn.Conv1d(2*lipsize, 2, kernel_size=1, groups=2, bias=False)

    def forward(self, stim):
        # visual stim -> spatial & temporal convolution:
        if self.separate:
            # [N, 1, T, 300, 300] -> [N, 1, T, 100, 100]
            conv_s = self.conv_s(stim)
            # [N, 1, T, 100, 100] -> [N, 2, t, 100, 100]
            conv_t = self.conv_t(conv_s)
            fire_lgn = self.fire_lgn(conv_t)
        else:
            curr_lgn = self.conv_0(stim)
            fire_lgn = self.fire_lgn(curr_lgn)

        # spatial convolution from LGN to V1 that should have direction selectivity
        # [N, 2, t, 100, 100] -> [N, 2, t, 50, 50]
        conv_v1 = self.conv_v1(fire_lgn)
        fire_v1 = self.fire_v1(conv_v1)

        # one more spatial convolution to make receptive field larger:
        # [N, 2, t, 50, 50] -> [N, 2, t/1, 20, 20]
        temp_mt = self.conv_mt(fire_v1)
        if self.pooling == 'mt':
            curr_mt = self.pool_mt(temp_mt)
        else:
            curr_mt = temp_mt
        fire_mt = self.fire_mt(curr_mt)

        # and then linear map to LIP
        # [N, 2, t/1, 20, 20] -> [N, 2x400, t/1, 1, 1] -> [N, 800]
        temp_lip = self.conv_m2l(fire_mt)
        if self.pooling == 'lip':
            curr_lip = self.pool_lip(temp_lip)
        else:
            curr_lip = temp_lip
        fire_lip = self.fire_lip(curr_lip)

        # collect for final decision
        # [N, 800, 1] -> [N, 2]
        predicts = self.decision(fire_lip.flatten(2)).squeeze()
        return predicts


class MotNet_Sep(BaseModel):
    '''
    Basic CNN model with the same structure of vismodel,
    separate connections from MT DS groups to LIP DS groups,
    see vismodel.py, rdmfull_simplified.py for comparison
    see also: MotNet_Conv, MotNet_Lin
    '''
    def __init__(
            self, separate=True, average=True, pooling='lip',
            ks=9, ss=3, kt=10, lipsize=400,
        ):
        super().__init__()
        self.separate = separate
        self.average = average
        self.pooling = pooling

        # firstly, the same conv_s and conv_t before LGN
        # input should be N x 1 x T x 300x300
        #              batch chan T    H   W
        if separate:
            self.conv_s = nn.Conv3d(1, 1, kernel_size=( 1,ks,ks), padding=(0,3,3), stride=(1,ss,ss), bias=False)
            # became N x 1 x T x 100 x 100 (or 50x50 for stride=6)
            self.conv_t = nn.Conv3d(1, 2, kernel_size=(kt, 1, 1), padding=(0,0,0), stride=(1,1,1))
            # became N x 2 x t x 100 x 100 (or 50x50)
        else:
            self.conv_0 = nn.Conv3d(1, 2, kernel_size=(kt,ks,ks), padding=(0,3,3), stride=(1,ss,ss))

        # then activity of LGN:
        self.fire_lgn = nn.ReLU()

        # then V1 and its activity (N x 2 x t x 50 x 50)
        self.conv_v1 = nn.Conv3d(2, 2, kernel_size=(1,3,3), padding=(0,1,1), stride=(1,6//ss,6//ss))
        self.fire_v1 = nn.ReLU()

        # then MT and its activity (N x 2 x t x 20 x 20)
        self.conv_mt = nn.Conv3d(2, 2, kernel_size=(1,11,11), padding=(0,0,0), stride=(1,2,2), groups=2)
        if pooling == 'mt':
            self.pool_mt = nn.AdaptiveAvgPool3d((1, None, None))
        self.fire_mt = nn.ReLU()

        # then LIP and activity (N x 2 x t x 20 x 20) -> (N x 2 x t/1 x num)
        self.conn_m2l = MulLinear((2, 400, lipsize), bias=True)
        if pooling == 'lip':
            self.pool_lip = nn.AdaptiveAvgPool2d((1, None))
        self.fire_lip = nn.ReLU()

        # finally the output layer
        if average:
            # Global Average Pooling (GAP):
            self.decision = nn.AdaptiveAvgPool2d((1,1))
        else:
            self.decision = MulLinear((2, lipsize, 1), bias=False)

    def forward(self, stim):
        # visual stim -> spatial & temporal convolution:
        if self.separate:
            # [N, 1, T, 300, 300] -> [N, 1, T, 100, 100]
            conv_s = self.conv_s(stim)
            # [N, 1, T, 100, 100] -> [N, 2, t, 100, 100]
            conv_t = self.conv_t(conv_s)
            fire_lgn = self.fire_lgn(conv_t)
        else:
            curr_lgn = self.conv_0(stim)
            fire_lgn = self.fire_lgn(curr_lgn)

        # spatial convolution from LGN to V1 that should have direction selectivity
        # [N, 2, t, 100, 100] -> [N, 2, t, 50, 50]
        conv_v1 = self.conv_v1(fire_lgn)
        fire_v1 = self.fire_v1(conv_v1)

        # one more spatial convolution to make receptive field larger:
        # [N, 2, t, 50, 50] -> [N, 2, t/1, 20, 20]
        temp_mt = self.conv_mt(fire_v1)
        if self.pooling == 'mt':
            curr_mt = self.pool_mt(temp_mt)
        else:
            curr_mt = temp_mt
        fire_mt = self.fire_mt(curr_mt)

        # and then linear map to LIP
        # [N, 2, t/1, 20, 20] -> [N, 2, t/1, 400] -> [N, 2, 1, num]
        temp_lip = self.conn_m2l(fire_mt.flatten(3))
        if self.pooling == 'lip':
            curr_lip = self.pool_lip(temp_lip)
        else:
            curr_lip = temp_lip
        fire_lip = self.fire_lip(curr_lip)

        # collect for final decision
        # [N, 2, 1, num] -> [N, 2]
        predicts = self.decision(fire_lip).flatten(1)
        return predicts


class MotNet_Mix(BaseModel):
    '''
    Basic CNN model for testing, mixed connection to LIP layer.
    conv_s + conv_t, or conv3d; linear or average pool;
    '''
    def __init__(
            self, separate=True, average=True, pooling='lip',
            ks=9, ss=3, kt=10, lipsize=400,
        ):
        super().__init__()
        self.separate = separate
        self.average = average
        self.pooling = pooling

        # firstly, the same conv_s and conv_t before LGN
        # input should be N x 1 x T x 300x300
        #              batch chan T    H   W
        if separate:
            self.conv_s = nn.Conv3d(1, 1, kernel_size=( 1,ks,ks), padding=(0,3,3), stride=(1,ss,ss), bias=False)
            # became N x 1 x T x 100 x 100 (or 50x50 for stride=6)
            self.conv_t = nn.Conv3d(1, 2, kernel_size=(kt, 1, 1), padding=(0,0,0), stride=(1,1,1))
            # became N x 2 x t x 100 x 100 (or 50x50)
        else:
            self.conv_0 = nn.Conv3d(1, 2, kernel_size=(kt,ks,ks), padding=(0,3,3), stride=(1,ss,ss))

        # then activity of LGN:
        self.fire_lgn = nn.ReLU()

        # then V1 and its activity (N x 2 x t x 50 x 50)
        self.conv_v1 = nn.Conv3d(2, 2, kernel_size=(1,3,3), padding=(0,1,1), stride=(1,6//ss,6//ss))
        self.fire_v1 = nn.ReLU()

        # then MT and its activity (N x 2 x t x 20 x 20) -> (N x 2 x 1 x 20 x 20)
        self.conv_mt = nn.Conv3d(2, 2, kernel_size=(1,11,11), padding=(0,0,0), stride=(1,2,2), groups=2)
        if pooling == 'mt':
            self.pool_mt = nn.AdaptiveAvgPool3d((1, None, None))
        self.fire_mt = nn.ReLU()

        # then LIP and activity (N x 2 x 1 x 20 x 20) -> (N x 800 x 1 x 1 x 1)
        # groups=1, such that all channels together, gives 800 results:
        self.conn_m2l = nn.Conv3d(2, 2*lipsize, kernel_size=(1,20,20))
        if pooling == 'lip':
            self.pool_lip = nn.AdaptiveAvgPool3d((1, None, None))
        self.fire_lip = nn.ReLU()

        # finally the output layer
        if average:
            self.decision = nn.AdaptiveAvgPool2d((2,1))
        else:
            self.decision = nn.Conv1d(2*lipsize, 2, kernel_size=1, bias=False)

    def forward(self, stim):
        # visual stim -> spatial & temporal convolution:
        if self.separate:
            # [N, 1, T, 300, 300] -> [N, 1, T, 100, 100]
            conv_s = self.conv_s(stim)
            # [N, 1, T, 100, 100] -> [N, 2, t, 100, 100]
            conv_t = self.conv_t(conv_s)
            fire_lgn = self.fire_lgn(conv_t)
        else:
            curr_lgn = self.conv_0(stim)
            fire_lgn = self.fire_lgn(curr_lgn)

        # spatial convolution from LGN to V1 that should have direction selectivity
        # [N, 2, t, 100, 100] -> [N, 2, t, 50, 50]
        conv_v1 = self.conv_v1(fire_lgn)
        fire_v1 = self.fire_v1(conv_v1)

        # one more spatial convolution to make receptive field larger:
        # [N, 2, t, 50, 50] -> [N, 2, t/1, 20, 20]
        temp_mt = self.conv_mt(fire_v1)
        if self.pooling == 'mt':
            curr_mt = self.pool_mt(temp_mt)
        else:
            curr_mt = temp_mt
        fire_mt = self.fire_mt(curr_mt)

        # and then linear map to LIP
        # [N, 2, t/1, 20, 20] -> [N, 2x400, t/1, 1, 1] -> [N, 800]
        temp_lip = self.conn_m2l(fire_mt)
        if self.pooling == 'lip':
            curr_lip = self.pool_lip(temp_lip)
        else:
            curr_lip = temp_lip
        fire_lip = self.fire_lip(curr_lip)

        # collect for final decision
        # [N, 800, 1] -> [N, 2]
        predicts = self.decision(fire_lip.flatten(2)).squeeze()
        return predicts


class MotNet_Lin(BaseModel):
    '''
    Basic CNN model for testing, mixed linear connection to LIP layer.
    conv_s + conv_t, or conv3d; linear or average pool;
    '''
    def __init__(
            self, separate=True, average=True, pooling='lip',
            ks=9, ss=3, kt=10, lipsize=400,
        ):
        super().__init__()
        self.separate = separate
        self.average = average
        self.pooling = pooling

        # firstly, the same conv_s and conv_t before LGN
        # input should be N x 1 x T x 300x300
        #              batch chan T    H   W
        if separate:
            self.conv_s = nn.Conv3d(1, 1, kernel_size=( 1,ks,ks), padding=(0,3,3), stride=(1,ss,ss), bias=False)
            # became N x 1 x T x 100 x 100 (or 50x50 for stride=6)
            self.conv_t = nn.Conv3d(1, 2, kernel_size=(kt, 1, 1), padding=(0,0,0), stride=(1,1,1))
            # became N x 2 x t x 100 x 100 (or 50x50)
        else:
            self.conv_0 = nn.Conv3d(1, 2, kernel_size=(kt,ks,ks), padding=(0,3,3), stride=(1,ss,ss))

        # then activity of LGN:
        self.fire_lgn = nn.ReLU()

        # then V1 and its activity (N x 2 x t x 50 x 50)
        self.conv_v1 = nn.Conv3d(2, 2, kernel_size=(1,3,3), padding=(0,1,1), stride=(1,6//ss,6//ss))
        self.fire_v1 = nn.ReLU()

        # then MT and its activity (N x 2 x t x 20 x 20) -> (N x 2 x t/1 x 20 x 20)
        self.conv_mt = nn.Conv3d(2, 2, kernel_size=(1,11,11), padding=(0,0,0), stride=(1,2,2), groups=2)
        if pooling == 'mt':
            self.pool_mt = nn.AdaptiveAvgPool3d((1, None, None))
        self.fire_mt = nn.ReLU()

        # then LIP and activity (N x 2 x t/1 x 20 x 20) -> (N x 1 x 800)
        self.conn_m2l = nn.Linear(800, 2*lipsize, bias=True)
        if pooling == 'lip':
            self.pool_lip = nn.AdaptiveAvgPool2d((1, None))
        self.fire_lip = nn.ReLU()

        # finally the output layer
        if average:
            self.decision = nn.AdaptiveAvgPool1d(2)
        else:
            self.decision = nn.Linear(2*lipsize, 2, bias=False)

    def forward(self, stim):
        # visual stim -> spatial & temporal convolution:
        if self.separate:
            # [N, 1, T, 300, 300] -> [N, 1, T, 100, 100]
            conv_s = self.conv_s(stim)
            # [N, 1, T, 100, 100] -> [N, 2, t, 100, 100]
            conv_t = self.conv_t(conv_s)
            fire_lgn = self.fire_lgn(conv_t)
        else:
            curr_lgn = self.conv_0(stim)
            fire_lgn = self.fire_lgn(curr_lgn)

        # spatial convolution from LGN to V1 that should have direction selectivity
        # [N, 2, t, 100, 100] -> [N, 2, t, 50, 50]
        conv_v1 = self.conv_v1(fire_lgn)
        fire_v1 = self.fire_v1(conv_v1)

        # one more spatial convolution to make receptive field larger:
        # [N, 2, t, 50, 50] -> [N, 2, t/1, 20, 20]
        temp_mt = self.conv_mt(fire_v1)
        if self.pooling == 'mt':
            curr_mt = self.pool_mt(temp_mt)
        else:
            curr_mt = temp_mt
        fire_mt = self.fire_mt(curr_mt)

        # and then linear map to LIP
        # [N, 2, 1, 20, 20] -> [N, t/1, 800]
        temp_lip = self.conn_m2l(fire_mt.transpose(1,2).flatten(2))
        if self.pooling == 'lip':
            curr_lip = self.pool_lip(temp_lip)
        else:
            curr_lip = temp_lip
        fire_lip = self.fire_lip(curr_lip)

        # collect for final decision
        # [N, 1, 800] -> [N, 2]
        predicts = self.decision(fire_lip.squeeze())
        return predicts


class MotNet_Mul(BaseModel):
    '''
    Basic CNN model with the same structure of vismodel,
    separate connections from MT DS groups to LIP DS groups,
    unfold, with different kernels at convolution.
    see also: MotNet_Sep
    '''
    def __init__(
            self, separate=True, average=True, pooling='lip',
            ks=9, ss=3, kt=10, lipsize=400, shared=False,
        ):
        super().__init__()
        self.separate = separate
        self.average = average
        self.pooling = pooling

        # firstly, the same conv_s and conv_t before LGN
        # input should be N x 1 x T x 300x300
        #              batch chan T    H   W
        if separate:
            self.conv_s = nn.Conv3d(1, 1, kernel_size=( 1,ks,ks), padding=(0,3,3), stride=(1,ss,ss), bias=False)
            # became N x 1 x T x 100 x 100 (or 50x50 for stride=6)
            self.conv_t = nn.Conv3d(1, 2, kernel_size=(kt, 1, 1), padding=(0,0,0), stride=(1,1,1))
            # became N x 2 x t x 100 x 100 (or 50x50)
        else:
            self.conv_0 = nn.Conv3d(1, 2, kernel_size=(kt,ks,ks), padding=(0,3,3), stride=(1,ss,ss))

        # then activity of LGN:
        self.fire_lgn = nn.ReLU()

        # then V1 and its activity (N x 2 x t x 50 x 50) -> (Nxt, 1, 2x3x3, 50x50)
        # self.conv_v1 = nn.Conv3d(2, 2, kernel_size=(1,3,3), padding=(0,1,1), stride=(1,6//ss,6//ss))
        self.expd_lgn = nn.Unfold(kernel_size=(3,3), padding=(1,1), stride=(6//ss,6//ss))
        # we need a weight matrix of shape [2, 2x3x3, 50x50]:
        self.conn_l2v = DotLinear((2, 2*3*3, 1 if shared else 50*50), reduce=-2, bias=True)
        self.fire_v1 = nn.ReLU()

        # then MT and its activity (Nxtx2 x 1 x 50 x 50) -> (Nxt, 2, 11x11, 20x20)
        # self.conv_mt = nn.Conv3d(2, 2, kernel_size=(1,11,11), padding=(0,0,0), stride=(1,2,2), groups=2)
        self.expd_v1 = nn.Unfold(kernel_size=(11,11), padding=(0,0), stride=(2,2))
        # we need a weight matrix of shape [2, 11x11, 20x20]:
        self.conn_v2m = DotLinear((2, 11*11, 1 if shared else 20*20), reduce=-2, bias=True)
        if pooling == 'mt':
            self.pool_mt = nn.AdaptiveAvgPool3d((1, None, None))
        self.fire_mt = nn.ReLU()

        # then LIP and activity (Nxt x 2 x 400) -> (Nxt x 2 x 400)
        self.conn_m2l = MulLinear((2, 400, lipsize), bias=True)
        if pooling == 'lip':
            self.pool_lip = nn.AdaptiveAvgPool3d((1, None, None))
        self.fire_lip = nn.ReLU()

        # finally the output layer
        if average:
            self.decision = nn.AdaptiveAvgPool3d((None,1,1))
        else:
            self.decision = MulLinear((2, lipsize, 1), bias=False)

    def forward(self, stim):
        # visual stim -> spatial & temporal convolution:
        if self.separate:
            # [N, 1, T, 300, 300] -> [N, 1, T, 100, 100]
            conv_s = self.conv_s(stim)
            # [N, 1, T, 100, 100] -> [N, 2, t, 100, 100]
            conv_t = self.conv_t(conv_s)
            fire_lgn = self.fire_lgn(conv_t)
        else:
            curr_lgn = self.conv_0(stim)
            fire_lgn = self.fire_lgn(curr_lgn)

        n, t = fire_lgn.shape[0], fire_lgn.shape[2]
        # spatial convolution by unfold+matmul from LGN to V1:
        # [N, 2, t, 100, 100] -> [N, t, 2, 100, 100] ->
        # [Nxt, 2, 100, 100] -> [Nxt, K1, O1]
        temp_lgn = fire_lgn.transpose(1,2).reshape(-1, 2, *fire_lgn.shape[-2:])
        expd_lgn = self.expd_lgn(temp_lgn)
        # [M, K1, O1] -> [M, 1, K1, O1] -> [M, C, 1, O1]
        conv_v1 = self.conn_l2v(expd_lgn.unsqueeze(1))
        fire_v1 = self.fire_v1(conv_v1)

        # one more spatial convolution to make receptive field larger:
        # [M, C, 1, O1] -> [MxC, 1, 50, 50] -> [MxC, K5, O5]
        expd_v1 = self.expd_v1(fire_v1.view(-1, 1, 50, 50))
        # [MxC, K5, O5] -> [M, C, K5, O5] -> [M, C, 1, O5]
        temp_mt = self.conn_v2m(expd_v1.view(-1, 2, *expd_v1.shape[-2:]))
        if self.pooling == 'mt':
            curr_mt = self.pool_mt(temp_mt.view(n, t, 2, -1)).view(n, t, 1, -1)
        else:
            curr_mt = temp_mt
        fire_mt = self.fire_mt(curr_mt)

        # and then linear map to LIP
        # [M, 2, 1, O5] -> [Nxt, 2, 1, 400]
        temp_lip = self.conn_m2l(fire_mt)
        # [N, t, 2, 400] -> [N, 1, 2, 400] -> [N, 2, 1, 400]
        if self.pooling == 'lip':
            curr_lip = self.pool_lip(temp_lip.view(n, t, 2, -1))
        else:
            curr_lip = temp_lip
        fire_lip = self.fire_lip(curr_lip).transpose(1,2)

        # collect for final decision
        # [N, 2, 1, 400] -> [N, 2]
        predicts = self.decision(fire_lip).flatten(1)
        return predicts


class MotNet_RNN(BaseModel):
    '''
    Basic CNN model for testing, LIP as an RNN layer.
    conv_s + conv_t, or conv3d; linear or average pool;
    '''
    def __init__(
            self, separate=True, average=True, rnn_bias=False,
            ks=9, ss=3, kt=10, lipsize=400,
        ):
        super().__init__()
        self.separate = separate
        self.average = average

        # firstly, the same conv_s and conv_t before LGN
        # input should be N x 1 x T x 300x300
        #              batch chan T    H   W
        if separate:
            self.conv_s = nn.Conv3d(1, 1, kernel_size=( 1,ks,ks), padding=(0,3,3), stride=(1,ss,ss), bias=False)
            # became N x 1 x T x 100 x 100 (or 50x50 for stride=6)
            self.conv_t = nn.Conv3d(1, 2, kernel_size=(kt, 1, 1), padding=(0,0,0), stride=(1,1,1))
            # became N x 2 x t x 100 x 100 (or 50x50)
        else:
            self.conv_0 = nn.Conv3d(1, 2, kernel_size=(kt,ks,ks), padding=(0,3,3), stride=(1,ss,ss))

        # then activity of LGN:
        self.fire_lgn = nn.ReLU()

        # then V1 and its activity (N x 2 x t x 50 x 50)
        self.conv_v1 = nn.Conv3d(2, 2, kernel_size=(1,3,3), padding=(0,1,1), stride=(1,6//ss,6//ss))
        self.fire_v1 = nn.ReLU()

        # then MT and its activity (N x 2 x t x 20 x 20)
        self.conv_mt = nn.Conv3d(2, 2, kernel_size=(1,11,11), padding=(0,0,0), stride=(1,2,2), groups=2)
        self.fire_mt = nn.ReLU()

        # then LIP as RNN layer: (N x 2 x t x 400) -> (N x 800)
        self.pops_lip = nn.RNN(800, 2*lipsize, nonlinearity='relu', bias=rnn_bias, batch_first=True)

        # finally the output layer
        # (N x 800) -> (N x 2)
        if average:
            self.decision = nn.AdaptiveAvgPool1d(2)
        else:
            self.decision = nn.Linear(2*lipsize, 2, bias=False)

    def forward(self, stim):
        # visual stim -> spatial & temporal convolution:
        if self.separate:
            # [N, 1, T, 300, 300] -> [N, 1, T, 100, 100]
            conv_s = self.conv_s(stim)
            # [N, 1, T, 100, 100] -> [N, 2, t, 100, 100]
            conv_t = self.conv_t(conv_s)
            fire_lgn = self.fire_lgn(conv_t)
        else:
            curr_lgn = self.conv_0(stim)
            fire_lgn = self.fire_lgn(curr_lgn)

        # spatial convolution from LGN to V1 that should have direction selectivity
        # [N, 2, t, 100, 100] -> [N, 2, t, 50, 50]
        conv_v1 = self.conv_v1(fire_lgn)
        fire_v1 = self.fire_v1(conv_v1)

        # one more spatial convolution to make receptive field larger:
        # [N, 2, t, 50, 50] -> [N, 2, t, 20, 20]
        conv_mt = self.conv_mt(fire_v1)
        fire_mt = self.fire_mt(conv_mt)

        # and then linear map to LIP
        # NOTE that RNN requires input be [N, t, C] in our case, so:
        # [N, 2, t, 20, 20] -> [N, t, 800]
        curr_lip = fire_mt.transpose(1,2).flatten(2)
        _, fire_lip = self.pops_lip(curr_lip)

        # collect for final decision
        # [1, N, 800] -> [N, 2]
        predicts = self.decision(fire_lip.squeeze())
        return predicts


class MotionRNN(BaseModel):
    '''
    Basic CNN+RNN model with the similar structure of vismodel,
    unfold, with different kernels at convolution.
    see vismodel.py, rdmfull_simplified.py for comparison.
    see also MotNet_Mul, MotNet_RNN
    '''
    def __init__(
            self, separate=True, average=True,
            ks=9, ss=3, kt=10, lipsize=400,
            shared=False, rnn_bias=False,
        ):
        super().__init__()
        self.separate = separate
        self.average = average
        self.shared = shared

        # firstly, the same conv_s and conv_t before LGN
        # input should be N x 1 x T x 300x300
        #              batch chan T    H   W
        if separate:
            self.conv_s = nn.Conv3d(1, 1, kernel_size=( 1,ks,ks), padding=(0,3,3), stride=(1,ss,ss), bias=False)
            # became N x 1 x T x 100 x 100 (or 50x50 for stride=6)
            self.conv_t = nn.Conv3d(1, 2, kernel_size=(kt, 1, 1), padding=(0,0,0), stride=(1,1,1))
            # became N x 2 x t x 100 x 100 (or 50x50)
        else:
            self.conv_0 = nn.Conv3d(1, 2, kernel_size=(kt,ks,ks), padding=(0,3,3), stride=(1,ss,ss))

        # then activity of LGN:
        self.fire_lgn = nn.ReLU()

        # then V1 and its activity (N x 2 x t x 50 x 50) -> (Nxt, 1, 2x3x3, 50x50)
        # self.conv_v1 = nn.Conv3d(2, 2, kernel_size=(1,3,3), padding=(0,1,1), stride=(1,6//ss,6//ss))
        self.expd_lgn = nn.Unfold(kernel_size=(3,3), padding=(1,1), stride=(6//ss,6//ss))
        # we need a weight matrix of shape [2, 2x3x3, 50x50]:
        self.conn_l2v = DotLinear((2, 2*3*3, 1 if shared else 50*50), reduce=-2, bias=True)
        self.fire_v1 = nn.ReLU()

        # then MT and its activity (Nxtx2 x 1 x 50 x 50) -> (Nxt, 2, 11x11, 20x20)
        # self.conv_mt = nn.Conv3d(2, 2, kernel_size=(1,11,11), padding=(0,0,0), stride=(1,2,2), groups=2)
        self.expd_v1 = nn.Unfold(kernel_size=(11,11), padding=(0,0), stride=(2,2))
        # we need a weight matrix of shape [2, 11x11, 20x20]:
        self.conn_v2m = DotLinear((2, 11*11, 1 if shared else 20*20), reduce=-2, bias=True)
        self.fire_mt = nn.ReLU()

        # then LIP as RNN layer: (N x 2 x t x 400) -> (N x 800)
        self.pops_lip = nn.RNN(800, 2*lipsize, nonlinearity='relu', bias=rnn_bias, batch_first=True)

        # finally the output layer: (N x 800) -> (N x 2)
        if average:
            self.decision = nn.AdaptiveAvgPool1d(2)
        else:
            self.decision = nn.Linear(2*lipsize, 2, bias=False)

    def from_conv(self, model:MotNet_RNN):
        # copy common params:
        state0 = self.state_dict()
        state1 = model.state_dict()
        for name in state0.keys():
            if name in state1:
                print(f'{name} <- {name}')
                state0[name].copy_(state1[name])

        # the conv kernel should be expanded:
        pairs = {
            'conn_l2v.weight':  'conv_v1.weight',
            'conn_l2v.bias':    'conv_v1.bias',
            'conn_v2m.weight':  'conv_mt.weight',
            'conn_v2m.bias':    'conv_mt.bias',
        }
        for dest, src in pairs.items():
            if dest in state0 and src in state1:
                print(f'{dest} <- {src}')
                state0[dest].copy_(state1[src].view(*state0[dest].shape[:-1], 1))

    def forward(self, stim):
        # visual stim -> spatial & temporal convolution:
        if self.separate:
            # [N, 1, T, 300, 300] -> [N, 1, T, 100, 100]
            conv_s = self.conv_s(stim)
            # [N, 1, T, 100, 100] -> [N, 2, t, 100, 100]
            conv_t = self.conv_t(conv_s)
            fire_lgn = self.fire_lgn(conv_t)
        else:
            curr_lgn = self.conv_0(stim)
            fire_lgn = self.fire_lgn(curr_lgn)

        n, t = fire_lgn.shape[0], fire_lgn.shape[2]
        # spatial convolution by unfold+matmul from LGN to V1:
        # [N, 2, t, 100, 100] -> [N, t, 2, 100, 100] ->
        # [Nxt, 2, 100, 100] -> [Nxt, K1, O1]
        temp_lgn = fire_lgn.transpose(1,2).reshape(-1, 2, *fire_lgn.shape[-2:])
        expd_lgn = self.expd_lgn(temp_lgn)
        # [M, K1, O1] -> [M, 1, K1, O1] -> [M, C, 1, O1]
        conv_v1 = self.conn_l2v(expd_lgn.unsqueeze(1))
        # conv_v1 = self.conv_v1(fire_lgn).transpose(1,2).reshape(-1,2,1,2500)
        # conv_v1 = self.conv_v1(fire_lgn)
        fire_v1 = self.fire_v1(conv_v1)

        # one more spatial convolution to make receptive field larger:
        # [M, C, 1, O1] -> [MxC, 1, 50, 50] -> [MxC, K5, O5]
        expd_v1 = self.expd_v1(fire_v1.view(-1, 1, 50, 50))
        # [MxC, K5, O5] -> [M, C, K5, O5] -> [M, C, 1, O5]
        curr_mt = self.conn_v2m(expd_v1.view(-1, 2, *expd_v1.shape[-2:]))
        # curr_mt = self.conv_mt(fire_v1.view(n,t,2,50,50).transpose(1,2)).transpose(1,2).reshape(-1,2,1,400)
        # curr_mt = self.conv_mt(fire_v1).transpose(1,2).reshape(-1,2,1,400)
        fire_mt = self.fire_mt(curr_mt)

        # and then linear map to LIP
        # [M, 2, 1, O5] -> [N, t, 2*400] -> [1, N, 800]
        curr_lip = fire_mt.view(n, t, -1)
        _, fire_lip = self.pops_lip(curr_lip)

        # collect for final decision
        # [1, N, 800] -> [N, 2]
        predicts = self.decision(fire_lip.squeeze())
        return predicts


class MotionNet(BaseModel):
    '''
    Basic CNN model with the similar structure of vismodel,
    unfold, with different kernels at convolution.
    see vismodel.py, rdmfull_simplified.py for comparison
    see also: MotNet_Mul, MotNet_Sep
    '''
    def __init__(
            self, separate=True, average=True,
            ks=9, ss=3, kt=10, lipsize=400, shared=False,
            disturb=None, prb=0, lvl=0,
        ):
        super().__init__()
        self.separate = separate
        self.average = average
        self.lipsize = lipsize
        self.shared = shared
        self.target = disturb
        self.params = [prb, lvl]

        # firstly, the same conv_s and conv_t before LGN
        # input should be N x 1 x T x 300x300
        #              batch chan T    H   W
        if separate:
            self.conv_s = nn.Conv3d(1, 1, kernel_size=( 1,ks,ks), padding=(0,3,3), stride=(1,ss,ss), bias=False)
            # became N x 1 x T x 100 x 100 (or 50x50 for stride=6)
            self.conv_t = nn.Conv3d(1, 2, kernel_size=(kt, 1, 1), padding=(0,0,0), stride=(1,1,1))
            # became N x 2 x t x 100 x 100 (or 50x50)
        else:
            self.conv_0 = nn.Conv3d(1, 2, kernel_size=(kt,ks,ks), padding=(0,3,3), stride=(1,ss,ss))

        # then activity of LGN:
        self.fire_lgn = nn.ReLU()

        # then V1 and its activity (N x 2 x t x 50 x 50) -> (Nxt, 1, 2x3x3, 50x50)
        # self.conv_v1 = nn.Conv3d(2, 2, kernel_size=(1,3,3), padding=(0,1,1), stride=(1,6//ss,6//ss))
        self.expd_lgn = nn.Unfold(kernel_size=(3,3), padding=(1,1), stride=(6//ss,6//ss))
        # we need a weight matrix of shape [2, 2x3x3, 50x50]:
        self.conn_l2v = DotLinear((2, 2*3*3, 1 if shared else 50*50), reduce=-2, bias=True)
        self.fire_v1 = nn.ReLU()

        # then MT and its activity (Nxtx2 x 1 x 50 x 50) -> (Nxt, 2, 11x11, 20x20)
        # self.conv_mt = nn.Conv3d(2, 2, kernel_size=(1,11,11), padding=(0,0,0), stride=(1,2,2), groups=2)
        self.expd_v1 = nn.Unfold(kernel_size=(11,11), padding=(0,0), stride=(2,2))
        # we need a weight matrix of shape [2, 11x11, 20x20]:
        self.conn_v2m = DotLinear((2, 11*11, 1 if shared else 20*20), reduce=-2, bias=True)
        self.fire_mt = nn.ReLU()

        # then LIP and activity (Nxt x 2 x 400) -> (Nxt x 2 x num)
        self.conn_m2l = MulLinear((2, 400, lipsize), bias=True)
        self.pool_lip = nn.AdaptiveAvgPool3d((1, None, None))
        self.fire_lip = nn.ReLU()

        # finally the output layer
        if average:
            self.decision = nn.AdaptiveAvgPool3d((None,1,1))
        else:
            self.decision = MulLinear((2, lipsize, 1), bias=False)

    def from_conv(self, model: MotNet_Sep):
        # copy common params:
        state0 = self.state_dict()
        state1 = model.state_dict()
        for name in state0.keys():
            if name in state1:
                print(f'{name:<16} <- {name}')
                state0[name].copy_(state1[name])

        # the conv kernel should be expanded:
        pairs = {
            'conn_l2v.weight':  'conv_v1.weight',
            'conn_l2v.bias':    'conv_v1.bias',
            'conn_v2m.weight':  'conv_mt.weight',
            'conn_v2m.bias':    'conv_mt.bias',
            'conn_m2l.weight':  'conv_m2l.weight',
            'conn_m2l.bias':    'conv_m2l.bias',
        }
        for dest, src in pairs.items():
            if dest in state0 and src in state1:
                print(f'{dest:<16} <- {src}')
                if 'm2l' in dest:
                    # our linear should be transposed to match the full-size conv3d
                    shape = list(state0[dest].shape)
                    shape[1], shape[2] = shape[2], shape[1]
                    state0[dest].copy_(state1[src].view(shape).transpose(1,2))
                else:
                    state0[dest].copy_(state1[src].view(*state0[dest].shape[:-1], 1))

    def disturb(self):
        """disturb model parameters (weight and/or bias)"""
        if self.target is None:
            return
        prob, level = self.params
        temp = self.target.split('-')
        device = next(self.parameters()).device

        if len(temp) == 1:
            # disturb should be applied to neuron (after relu)
            if self.target == 'lgn':
                print('Disturb neuron: LGN')
                self.mask_lgn = torch.rand(1,2,1,100,100, device=device) < abs(prob)
            elif self.target == 'v1':
                print('Disturb neuron: V1')
                self.mask_v1 = torch.rand(1,2,1,50*50, device=device) < abs(prob)
            elif self.target == 'mt':
                print('Disturb neuron: MT')
                self.mask_mt = torch.rand(1,2,1,20*20, device=device) < abs(prob)
            elif self.target == 'lip':
                print('Disturb neuron: LIP')
                self.mask_lip = torch.rand(1,2,1,self.lipsize, device=device) < abs(prob)
            return

        layer = temp[0]
        target = temp[1]
        module = getattr(self, layer)

        if target == 'all' or target == 'weight':
            print(f'Disturb {layer}.weight')
            disturb_inplace(module.weight.data, prob, level)
        if target == 'all' or target == 'bias':
            if module.bias is not None:
                print(f'Disturb {layer}.bias')
                disturb_inplace(module.bias.data, prob, level)

    def forward(self, stim):
        prob, level = self.params

        # visual stim -> spatial & temporal convolution:
        if self.separate:
            # [N, 1, T, 300, 300] -> [N, 1, T, 100, 100]
            conv_s = self.conv_s(stim)
            # [N, 1, T, 100, 100] -> [N, 2, t, 100, 100]
            conv_t = self.conv_t(conv_s)
            curr_lgn = conv_t
        else:
            curr_lgn = self.conv_0(stim)

        if hasattr(self, 'mask_lgn'):
            disturb_inplace(curr_lgn, prob, level, self.mask_lgn)
        fire_lgn = self.fire_lgn(curr_lgn)

        n, t = fire_lgn.shape[0], fire_lgn.shape[2]
        # spatial convolution by unfold+matmul from LGN to V1:
        # [N, 2, t, 100, 100] -> [N, t, 2, 100, 100] ->
        # [Nxt, 2, 100, 100] -> [Nxt, K1, O1]
        temp_lgn = fire_lgn.transpose(1,2).reshape(-1, 2, *fire_lgn.shape[-2:])
        expd_lgn = self.expd_lgn(temp_lgn)
        # [M, K1, O1] -> [M, 1, K1, O1] -> [M, C, 1, O1]
        curr_v1 = self.conn_l2v(expd_lgn.unsqueeze(1))

        if hasattr(self, 'mask_v1'):
            disturb_inplace(curr_v1, prob, level, self.mask_v1)
        fire_v1 = self.fire_v1(curr_v1)

        # one more spatial convolution to make receptive field larger:
        # [M, C, 1, O1] -> [MxC, 1, 50, 50] -> [MxC, K5, O5]
        expd_v1 = self.expd_v1(fire_v1.view(-1, 1, 50, 50))
        # [MxC, K5, O5] -> [M, C, K5, O5] -> [M, C, 1, O5]
        curr_mt = self.conn_v2m(expd_v1.view(-1, 2, *expd_v1.shape[-2:]))

        if hasattr(self, 'mask_mt'):
            disturb_inplace(curr_mt, prob, level, self.mask_mt)
        fire_mt = self.fire_mt(curr_mt)

        # and then linear map to LIP
        # [M, 2, 1, O5] -> [Nxt, 2, 1, 400]
        temp_lip = self.conn_m2l(fire_mt)
        # [N, t, 2, 400] -> [N, 1, 2, 400] -> [N, 2, 1, 400]
        curr_lip = self.pool_lip(temp_lip.view(n, t, 2, -1))
        curr_lip = curr_lip.transpose(1,2)

        if hasattr(self, 'mask_lip'):
            disturb_inplace(curr_lip, prob, level, self.mask_lip)
        fire_lip = self.fire_lip(curr_lip)

        # collect for final decision
        # [N, 2, 1, 400] -> [N, 2]
        predicts = self.decision(fire_lip).flatten(1)
        return predicts


def _test(MotNet, avg=True, sep=True):
    from tqdm.auto import trange
    mdl = MotNet(separate=sep, average=avg)
    # mdl = MotionNet()
    mdl.info(True)
    for ii in trange(20, ncols=90):
        im0 = torch.rand((8,1,20,300,300))
        prd = mdl(im0)
    print(prd.shape)
    print(prd)


def _test_profile(MotNet):
    from line_profiler import LineProfiler
    lp = LineProfiler()
    lw = lp(_test)
    lp.add_function(MotNet.forward)
    lw(MotNet, True)
    lp.print_stats()


def _test_compare(n=100, device='cpu'):
    import time
    import numpy as np
    from tqdm.auto import trange

    t0 = time.time()
    # mdl1 = MotNet_RNN(True, False).to(device)
    # # mdl2 = MotionRNN(True, False, shared=True).to(device)
    # mdl2 = MotionRNN(True, False, shared=False).to(device)
    mdl1 = MotNet_Conv(True, False).to(device)
    # mdl2 = MotionNet(True, False, shared=True).to(device)
    mdl2 = MotionNet(True, False, shared=False).to(device)
    mdl2.from_conv(mdl1)
    t1 = time.time() - t0

    mdl1.info(True)
    mdl2.info(True)

    diffs = np.zeros(n)
    t0 = time.time()
    for ii in trange(n):
        stim = torch.rand((8,1,20,300,300), device=device)
        prd1 = mdl1(stim)
        prd2 = mdl2(stim)
        # print(torch.hstack([prd1, prd2]))
        # print(prd1 - prd2)
        diffs[ii] = (prd1 - prd2).abs().max().item()
    # print(diffs)
    t2 = time.time() - t0
    print('%d: %.3g +/- %.3g, max=%.3g, costs %.2fs+%.2fs' % (
        n, np.mean(diffs), np.std(diffs), np.max(diffs), t1, t2,
        ))
    return diffs, t1, t2


def _test_plot_comp(device='cuda', fname=None):
    import matplotlib.pyplot as plt
    # _test_compare(10, device)
    nums = [100, 100, 100, 200, 400, 800]
    delta = []
    tinit = []
    tpred = []
    for ii in nums:
        dd, t0, t1 = _test_compare(ii, device)
        delta.append(dd)
        tinit.append(t0)
        tpred.append(t1)
    fig, ax = plt.subplots(ncols=2, figsize=(8,4))
    ax[0].boxplot(delta)
    # ax[0].violinplot(delta)
    ax[0].set_xticks(range(1, len(nums)+1), labels=nums)
    ax[1].plot([x+y for x, y in zip(tpred, tinit)], 'c')
    ax[1].plot(tpred, 'r')
    ax[1].set_xticks(range(len(nums)), labels=nums)
    # ax2 = ax[1].twinx()
    # ax2.plot(tinit, 'c')
    if fname is None:
        fname = f'{device}-comp.png'
    fig.savefig(fname)


def _test_disturb(target, prb=1, lvl=0):
    mdl = MotionNet(disturb=target, prb=prb, lvl=lvl)
    mdl.info(True)

    img = torch.rand((8,1,20,300,300))
    pr1 = mdl(img)
    mdl.disturb()
    pr2 = mdl(img)

    print(torch.hstack([pr1, pr2]))
    print(pr1 - pr2)


if __name__ == '__main__':
    # _test(MotNet_Sep, False)
    # _test(MotNet_Mix, False)
    # _test(MotNet_Lin, False)
    # _test(MotNet_RNN, False)
    # _test(MotNet_Mul, False)
    # _test(MotionNet, False)
    # _test_profile(MotNet_RNN)
    # _test_compare(100, 'cuda')
    _test_plot_comp('cuda', 'comp2.png')
    # _test_disturb('conn_l2v-weight', -1, 0)
    # _test_disturb('lip', -1, 0)
