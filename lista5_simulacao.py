# -*- coding: utf-8 -*-
"""
EST5104 - Inferência Bayesiana - 2025/1
@author: Cleiton Moya de Almeida
Prof. Josemar Rodrigues
Lista 5 - Power Prior - Questão 2 (Simulação)
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm

plt.rcParams.update({'font.size': 8, 'axes.titlesize': 8})
rng = np.random.default_rng(seed=42)

# parâmetros conhecidos
theta = theta0 = 5
n = 100
n0 = 80

# Simulação
N = 1000
a0_ = np.arange(0.01,1.01,0.01)
Prob = []

params = ((2,2),(2,4),(4,2))

for j,p in enumerate(params):
    sigma0,sigma = p
    
    prob_p = []
    for a0 in a0_:
        
        n_sucessos = 0
        for _ in range(N):
            
            # Simula os dados históricos
            y0 = norm.rvs(loc=theta0, scale=sigma0, size=n0, random_state=rng)
            y0_mean = y0.mean()
            tau0 = n0/sigma0**2
        
            # Simula os dados atuais
            y = norm.rvs(loc=theta, scale=sigma, size=n, random_state=rng)
            y_mean = y.mean()
            tau = n/sigma**2
        
            # Distribuição a posteriori
            mean_star = (tau*y_mean + a0*tau0*y0_mean)/(tau+ a0*tau0)
            sigma2_star = 1/(tau+a0*tau0)
            sigma_star = np.sqrt(sigma2_star)
            
            # Intervalo de credibilidade
            post_pdf = norm(loc=mean_star, scale=sigma_star)
            a,b = post_pdf.interval(confidence=0.95)
            
            if (theta >= a) and (theta <= b):
                n_sucessos = n_sucessos + 1
        prob_p.append(n_sucessos/N)
    Prob.append(prob_p)


#%% Gráfico - Probabilidade de cobertura
fig,ax = plt.subplots(figsize=(6,2.5), layout="constrained")
ax.axhline(y=0.95, linestyle='--', color='gray', label="95%")
ax.plot(a0_, Prob[0], label=r"$\sigma_0^2=4, \sigma^2=4$")
ax.plot(a0_, Prob[1], label=r"$\sigma_0^2=4, \sigma^2=16$")
ax.plot(a0_, Prob[2], label=r"$\sigma_0^2=16, \sigma^2=4$")
ax.set_ylim(0.5,1.09)
ax.set_xlabel(r"$a_0$")
ax.set_ylabel("Prob. de cobertura")
ax.legend(loc="lower right")