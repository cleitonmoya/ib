# -*- coding: utf-8 -*-
"""
EST5104 - Inferência Bayesiana - 2025/1
@author: Cleiton Moya de Almeida
Prof. Josemar Rodrigues
Avaliação 2 - Questão 1c - PCP
"""

import numpy as np
from numpy import sqrt, exp, log
import matplotlib.pyplot as plt


plt.rcParams.update({'font.size': 8, 'axes.titlesize': 8})
rng = np.random.default_rng(seed=42)

def kl(theta, theta0):
    kl = theta*log(theta/theta0) + (1-theta)*log((1-theta)/(1-theta0))
    return kl

def dist(theta, theta0):
    d = sqrt(kl(theta, theta0))
    return d

# PC Prior
def pcp(theta, theta0, lamb):
    d = dist(theta, theta0)
    y1 = (1/d)*lamb*exp(-lamb*d)
    y2 = abs(log(theta/theta0)-log((1-theta)/(1-theta0)))
    y = y1*y2
    return y

# Gráfico para diferentes taxas de penalização
theta0 = 0.5
theta_range = np.arange(0.1,1.0,0.01)
lamb_range = np.arange(0,5.01,0.01)

####
fig,ax = plt.subplots(figsize=(3,2.5), layout="constrained")
lamb_range2 = [0.2, 0.5, 0.7, 1, 2]
Y = []
for lamb in lamb_range2:
    y = [pcp(theta, theta0, lamb) for theta in theta_range]
    ax.plot(theta_range, y, label=fr'$\lambda={lamb}$')
ax.legend()
ax.set_xlabel(r'$\theta$')
ax.set_ylabel(r'$\pi(\theta)$')
ax.set_xticks(np.arange(0,1.1,0.1))
