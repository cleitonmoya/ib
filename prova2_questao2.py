# -*- coding: utf-8 -*-
"""
EST5104 - Inferência Bayesiana - 2025/1
@author: Cleiton Moya de Almeida
Prof. Josemar Rodrigues
Lista 5 - Power Prior - Questão 2 (Simulação)
"""

import numpy as np
import matplotlib.pyplot as plt

plt.rcParams.update({'font.size': 8, 'axes.titlesize': 8})

np.random.seed(5)

# Parâmetro gerado pela natureza (desconhecido)
theta = 5 

# Dados atuais
n = 1
sigma2 = 1

# Dados históricos
tau2 = 1            # variância para gerar theta_i
n0 = 3              # tamanho da amostra de dados históricos 

# Parâmetros da simulação
N = 10000  # número de simulações
a0_range = np.arange(0.01,1.01,0.01)


Prob = [] # Array com probabilidade de cobertura para cada a0
for a0 in a0_range:
    
    n_sucessos = 0
    for nit in range(N):
        
        # Dados históricos - modelo hierárquico com priori Bayes-Laplace para theta
        theta_i = np.random.normal(loc=theta, scale=np.sqrt(tau2), size=n0)
        x0 = np.random.normal(loc=theta_i, scale=np.sqrt(sigma2))
        x0_mean = np.mean(x0)  # Média dos dados históricos
        
        # Simulação dos dados atuais
        x = np.random.normal(loc=theta, scale=np.sqrt(sigma2), size=n)

        # Distribuição a posteriori (dada a power prior)
        mu_post = (3*a0*x0_mean + 2*x)/(3*a0 + 2)
        sigma2_post = 2/(3*a0 + 2)
        sigma_post = np.sqrt(sigma2_post)
        
        # Intervalo de credibilidade
        a = mu_post - 1.96*sigma_post
        b = mu_post + 1.96*sigma_post

        if a <= theta <= b:
            n_sucessos = n_sucessos + 1
    
    Prob.append(n_sucessos/N)


#%% Gráfico - Probabilidade de cobertura
fig,ax = plt.subplots(figsize=(6,2.5), layout="constrained")
ax.axhline(y=0.95, linestyle='--', color='red', label="95%")
ax.plot(a0_range, Prob)
ax.set_xlabel(r"$a_0$")
ax.set_ylabel("Prob. de cobertura")
ax.legend(loc="lower right")