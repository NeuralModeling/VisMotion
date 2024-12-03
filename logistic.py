#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Copyright (C) 2023-2024, 类生智能, all rights reserved.
'''
----------------------------------
@Modified By:  SU Jie
@Modified:     2024-04-17 10:49:59
----------------------------------
@File:         logistic.py
@Version:      1.0.0
@Created:      2023-06-15 09:46:54
@Author:       SU Jie
@Description:  Perform logistic regression as like glmfit in matlab
'''

import numpy as np


def logistic(x, b, k):
    # logistic for curve fitting
    return 1 / (1 + np.exp(-k*x-b))

def link(mu):
    # logit
    return np.log(mu / (1-mu))

def dlink(mu):
    # d logit
    return 1 / (mu * (1-mu))

def ilink(eta):
    # logistic with constrain
    lb = np.log(np.finfo(eta.dtype).eps)
    eta[eta<lb] = lb
    eta[eta>-lb] = -lb
    return 1 / (1 + np.exp(-eta))


def psychval(beta, x):
    return logistic(x, *beta)


def psychfit(x, p):
    """simplified glmfit (logistic) adapted from matlab

    Args:
        x (array): dependent variable
        p (array): the choice ratio

    Returns:
        beta (array of shape (2,)): the coeffient of logistic function
        se (array of shape (2,)): standard error w.r.t beta
    """
    # add const term to x
    xx = np.stack([1+0*x, x])
    # initialize as in matlab's glmfit
    eps = np.finfo(p.dtype).eps
    seps = np.sqrt(eps)
    nits = 100
    crit = 1e-6
    mu = (p + 0.5) / 2
    eta = link(mu)

    b = [0, 1]
    for ii in range(nits):
        deta = dlink(mu)
        z = eta + (p - mu) * deta

        sqrtvar = np.sqrt(mu) * np.sqrt(1-mu)
        sqrtirls = np.abs(deta) * sqrtvar
        sqrtw = 1 / sqrtirls

        wtol = np.max(sqrtw) * eps ** (2/3)
        t = sqrtw < wtol
        if np.any(t):
            t = t & (sqrtw != 0)
            if np.any(t):
                sqrtw[t] = wtol

        b_old = b
        # b, R = wfit(z, xx, sqrtw)
        yw = z * sqrtw
        xw = xx * sqrtw
        Q, R = np.linalg.qr(xw.T)
        b = np.linalg.lstsq(R, Q.T @ yw, rcond=None)[0]
        eta = xx.T @ b
        mu = ilink(eta)

        # if np.all(np.abs(b - b_old) <= crit * np.max(seps, np.abs(b_old))):
        if np.all(np.abs(b - b_old) <= crit * seps):
            break

    RI = np.linalg.lstsq(R, np.eye(xx.shape[0]), rcond=None)[0]
    C = RI @ RI.T
    se = np.sqrt(np.diag(C)).squeeze()
    return b, se


def _test_compare():
    import matplotlib.pyplot as plt

    from sklearn.linear_model import LogisticRegression
    from scipy.optimize import curve_fit
    from scipy.io import loadmat

    # data = loadmat('test_/tests-01.mat')
    # coherence = data['coherence'][0]
    # direction = data['direction'][0]
    # resp = data['resp'][0]
    res = np.load('test_rdm.npy')
    coherence = res[:, 0]
    direction = res[:, 1] - 1
    resp = res[:, 2] - 1

    dcoh = coherence * (2*direction-1)
    xx = np.unique(dcoh)
    r1 = 0*xx
    r2 = 0*xx
    ru = 0*xx

    for ii, cc in enumerate(xx):
        ix = dcoh == cc
        r1[ii] = np.mean(resp[ix]==0)
        r2[ii] = np.mean(resp[ix]==1)
        ru[ii] = np.mean(resp[ix]==2)

    yy = (r2 + 1 - r1)/2
    fig, ax = plt.subplots()
    ax.axvline(color='gray')
    ax.axhline(0.5, color='gray')
    ax.plot(xx/100, yy, 'k.')

    # curve fitting:
    popt, pcov = curve_fit(logistic, xx/100, yy)
    print(popt[0], popt[1])
    yfit = logistic(xx/100, *popt)
    ax.plot(xx/100, yfit, 'r')

    # logistic regression:
    mdl = LogisticRegression(penalty=None).fit(dcoh[...,np.newaxis]/100, resp)
    k = mdl.coef_.item()
    b = mdl.intercept_.item()
    print(b, k)
    prb = mdl.predict_proba(xx[...,np.newaxis]/100)[:,1]
    ax.plot(xx/100, prb, 'b')

    # glmfit:
    beta, se = psychfit(xx/100, yy)
    print(beta, se)
    pfit = logistic(xx/100, *beta)
    ax.plot(xx/100, pfit, 'm')

    # glmfit direct
    beta0, se0 = psychfit(dcoh/100, resp)
    print(beta0, se0)

    # glmfit: boundary
    beta, se = psychfit(xx/100, 0*yy)
    print(beta, se)
    pass


def _test_se():
    import pandas as pd
    import seaborn as sns
    x = np.array([
        2100, 2300, 2500, 2700, 2900, 3100,
        3300, 3500, 3700, 3900, 4100, 4300,
        ])
    n = np.array([
        48, 42, 31, 34, 31, 21, 23, 23, 21, 16, 17, 21,
        ])
    y = np.array([
        1, 2, 0, 3, 8, 8, 14, 17, 19, 15, 17, 21,
        ])

    b, se = psychfit(x, y/n)
    print('(%.9f, %.9f) +- (%.9f, %.9f)' % (*b, *se))

    data = pd.DataFrame()
    data['x'] = x
    data['p'] = y / n
    sns.regplot(x='x', y='p', data=data, logistic=True)
    pass



if __name__ == '__main__':
    _test_compare()
    # _test_se()
