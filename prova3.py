# -*- coding: utf-8 -*-
"""
EST5104 - Inferência Bayesiana - 2025/1
@author: Cleiton Moya de Almeida
Prof. Josemar Rodrigues
Avaliação 3 - Posterior Predictive Checking
"""

import numpy as np
import matplotlib.pyplot as plt
import warnings

# interrompe a execução em caso de alerta de erro numérico
warnings.filterwarnings('error')

plt.rcParams.update({'font.size': 8, 'axes.titlesize': 8, 'figure.titlesize': 10})
rng = np.random.default_rng(seed=42)

x = np.array([1, 1, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0])
N = len(x)

x1 = x.sum()
x0 = N-x1
a = x1+1
b = x0+1
theta_hat = x1/N

# Estatísticas
def T1(x):
    T1 = (x[:-1]!=x[1:]).sum()
    return T1

def T2(x):
    mean_x = x.mean()
    Sx = x.std(ddof=1)
    if mean_x > 0:
        T2 = Sx/mean_x
    else:
        T2 = np.nan
    return T2


def T3(x, theta):
    mu = theta
    var = theta*(1-theta)  # Var[x_i|theta_hat]
    if var>0:
        T3 = np.array([(xi-mu)**2 for xi in x]).sum()/var
    else:
        T3 = np.nan
    return T3

def amostraModeloBN(k):
    X = []
    c = 0
    while c<k:
        x = rng.binomial(n=1, p=theta)
        X.append(x)
        if x==0:
            c=c+1
    return np.array(X)

# Estatística esperada para a amostra
t1_obs = T1(x)
t2_obs = T2(x)
t3_obs = T3(x, theta_hat)

# Simulação
nc1b = nc2b = nc3b = 0
nc1nb = nc2nb = nc3nb = 0
T1b = []
T2b = []
T3b = []

T1nb = []
T2nb = []
T3nb = []

nsim = 100000
for _ in range(nsim):
    
    # Amostra theta (distribuição a posteriori - Beta)
    theta = rng.beta(a, b)
    
    # Amostra os dados replicados
    x_rep_b = rng.binomial(n=1, p=theta, size=N)  # modelo Binomial
    x_rep_nb = amostraModeloBN(x0)                # modelo Binomial-Negativa
    
    # Calcula as as grandezas de teste
    t1b = T1(x_rep_b)
    t2b = T2(x_rep_b)
    t3b = T3(x_rep_b, theta)
    
    t1nb = T1(x_rep_nb)
    t2nb = T2(x_rep_nb)
    t3nb = T3(x_rep_nb, theta)
    
    T1b.append(t1b)
    T2b.append(t2b)
    T3b.append(t3b)
    
    T1nb.append(t1nb)
    T2nb.append(t2nb)
    T3nb.append(t3nb)
    
    # Verifica se as grandezas calculadas extrapolaram as observadas
    if t1b >= t1_obs:
        nc1b = nc1b + 1
    if t2b >= t2_obs:
        nc2b = nc2b + 1  
    if t3b >= t3_obs:
        nc3b = nc3b + 1

    if t1nb >= t1_obs:
        nc1nb = nc1nb + 1
    if t2nb >= t2_obs:
        nc2nb = nc2nb + 1  
    if t3nb >= t3_obs:
        nc3nb = nc3nb + 1

# Cálculo dos p-valores
p_valor1b = nc1b/nsim
p_valor2b = nc2b/nsim
p_valor3b = nc3b/nsim

p_valor1nb = nc1nb/nsim
p_valor2nb = nc2nb/nsim
p_valor3nb = nc3nb/nsim

#%% Gráficos
# Modelo binomial
fig,ax = plt.subplots(ncols=3, layout='constrained', figsize=(5,2))
fig.suptitle('Modelo: Binomial')
ax[0].hist(T1b, edgecolor='w', alpha=0.5, bins=15, density=True)
ax[0].set_xticks([0,5,10,15,20])
ax[0].axvline(x=t1_obs, color='red', label=r'$T_1(\mathbf{x}_{\text{obs}})$')
ax[0].set_xlabel(r'$T_1(\mathbf{x}^\text{rep})$')
ax[0].legend()

ax[1].hist(T2b, edgecolor='w', alpha=0.5, bins=15, density=True)
ax[1].axvline(x=t2_obs, color='red', label=r'$T_2(\mathbf{x}_{\text{obs}})$')
ax[1].set_xlabel(r'$T_2(\mathbf{x}^\text{rep})$')
ax[1].legend()

ax[2].hist(T3b, edgecolor='w', alpha=0.5, bins=15, density=True)
ax[2].axvline(x=t3_obs, color='red', label=r'$T_3(\mathbf{x}_{\text{obs}}, \hat{\theta})$')
ax[2].set_xlabel(r'$T_3(\mathbf{x}^\text{rep}, \theta)$')
ax[2].legend()

# Modelo Binomial-Negativa
fig,ax = plt.subplots(ncols=3, layout='constrained', figsize=(5,2))
fig.suptitle('Modelo: Binomial-Negativa')
ax[0].hist(T1nb, edgecolor='w', alpha=0.5, bins=15, density=True)
ax[0].set_xticks([0,5,10,15,20])
ax[0].axvline(x=t1_obs, color='red', label=r'$T_1(\mathbf{x}_{\text{obs}})$')
ax[0].set_xlabel(r'$T_1(\mathbf{x}^\text{rep})$')
ax[0].legend()

ax[1].hist(T2nb, edgecolor='w', alpha=0.5, bins=15, density=True)
ax[1].axvline(x=t2_obs, color='red', label=r'$T_2(\mathbf{x}_{\text{obs}})$')
ax[1].set_xlabel(r'$T_2(\mathbf{x}^\text{rep})$')
ax[1].legend()

ax[2].hist(T3nb, edgecolor='w', alpha=0.5, bins=15, density=True)
ax[2].axvline(x=t3_obs, color='red', label=r'$T_3(\mathbf{x}_{\text{obs}}, \hat{\theta})$')
ax[2].set_xlabel(r'$T_3(\mathbf{x}^\text{rep}, \theta)$')
ax[2].legend()