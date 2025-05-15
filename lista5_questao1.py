# -*- coding: utf-8 -*-
"""
EST5104 - Inferência Bayesiana - 2025/1
@author: Cleiton Moya de Almeida
Prof. Josemar Rodrigues
Lista 5 - Power Prior - Questão 1 (Exemplo)
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm

plt.rcParams.update({'font.size': 8, 'axes.titlesize': 8})
rng = np.random.default_rng(seed=42)

# parâmetros conhecidos
sigma0 = 2
sigma = 4
theta = theta0 = 5
n = 100
n0 = 80
x_ = np.arange(3,7.01,0.01)

# Dados históricos
y0 = norm.rvs(loc=theta0, scale=sigma0, size=n0, random_state=rng)
y0_mean = y0.mean()
tau0 = n0/sigma0**2
y0_pdf = np.array([norm.pdf(x, loc=y0_mean, scale=sigma0/np.sqrt(n0)) for x in x_])

# Dados atuais
y = norm.rvs(loc=theta, scale=sigma, size=n, random_state=rng)
y_mean = y.mean()
tau = n/sigma**2
y_pdf = np.array([norm.pdf(x, loc=y_mean, scale=sigma/np.sqrt(n)) for x in x_])

#%%
fig,ax = plt.subplots(nrows=1, ncols=3, figsize=(7,2.5), layout="constrained",
                      sharey=True)
a0_ = (0.1, 0.5, 0.9) # hiperparâmetros da Power Prior
for i,a0 in enumerate(a0_):
    
    # Distribuição a posteriori
    mean_star = (tau*y_mean + a0*tau0*y0_mean)/(tau+ a0*tau0)
    sigma2_star = 1/(tau+a0*tau0)
    sigma_star = np.sqrt(sigma2_star)
    y_star_pdf = np.array([norm.pdf(x, loc=mean_star, scale=sigma_star) for x in x_])
    
    ax[i].set_title(fr"$a_0={a0}$")
    ax[i].plot(x_, y0_pdf, label=r"$\mathbf{y}_0$", linestyle="--")
    ax[i].plot(x_, y_pdf, label=r"$\mathbf{y}$", color="C0")
    ax[i].plot(x_, y_star_pdf, color="r",
            label=r"$\pi(\theta|\mathbf{y}, \mathbf{y}_0, a_0)$", )
    ax[i].set_box_aspect(1)
ax[0].legend()