# -*- coding: utf-8 -*-
"""
EST5104 - Inferência Bayesiana - 2025/1
@author: Cleiton Moya de Almeida
Prof. Josemar Rodrigues
Avaliação 2 - Questão 1d - Penalised Complexety Prior (PCP)
Modelo: X_i ~ Binomial(0.35), i=1,...,300
Priori: PCP
Modelo de base: Binomial(0.5)
Posteriori: Simulação via Metropolis-Hastings com Passeio Aleatório
"""
import numpy as np
from numpy import sqrt, exp, log
from scipy.stats import norm, uniform
import matplotlib.pyplot as plt
import seaborn as sns
import warnings

# interrompe a execução em caso de alerta de erro numérico
warnings.filterwarnings('error')

plt.rcParams.update({'font.size': 8, 'axes.titlesize': 8})
rng = np.random.default_rng(seed=42) # semente da simulação

# Divegência de Kullback-Leibler para binomial(theta)||binomial(theta0)
def kl(theta, theta0):
    kl = theta*(log(theta)-log(theta0)) + (1-theta)*(log((1-theta))-log((1-theta0)))
    return kl

def dist(theta, theta0):
    d = sqrt(kl(theta, theta0))
    return d

# PC Prior
# note que se theta=theta0, d=0 e PCP não fica bem definida. 
def pcp(theta, theta0, lamb):
    d = dist(theta, theta0)
    y1 = (1/d)*lamb*exp(-lamb*d)
    y2 = abs(log(theta)-log(theta0)-log(1-theta)+log(1-theta0))
    y = y1*y2
    return y

# Posterior
def pdf_des(S, N, theta, theta0, lamb):
    pi = pcp(theta, theta0, lamb)
    lik = theta**S*(1-theta)**(N-S)
    y = pi*lik
    return y

# Valor proposto de theta via Passeio Aleatório normal
def simula_theta_prop(mu,s):
    theta = norm.rvs(loc=mu, scale=s, random_state=rng)
    return theta

# Probabilidade de aceitação do valor proposto
def prob_aceit(theta, theta_prop, S, N, theta0, lamb):
    p1 = pdf_des(S, N, theta_prop, theta0, lamb)
    p2 = pdf_des(S, N, theta, theta0, lamb)
    if p2 > 0:
        alpha = min([1, p1/p2])
    else:
        alpha = 1
    return alpha
    
# Simulaçao
n_sim = 1100
burn = 100
N = 300
S = 105
theta0 = 0.5

# Distribuição proposta - passeio aleatório
mu = 0.5 
s = 0.2
lamb_range = [1, 10, 50, 100]

for lamb in lamb_range:
    theta = 0.7 # inicialização
    Theta = [theta]
    na = 0
    for n in range(1,n_sim):
        
        # simula o valor proposto
        theta_prop = simula_theta_prop(theta, s)
            
        # probabilidade de aceitação
        # (note que a PCP não é bem definida para theta=theta0)
        if (theta_prop > 0) and (theta_prop < 1) and (theta_prop !=theta0):
            alpha = prob_aceit(theta ,theta_prop, S, N, theta0, lamb)
        else:
            alpha = 0
        
        # critério de aceição
        u = uniform.rvs(0,1, random_state=rng)
        if  u < alpha:
            theta = theta_prop
            na = na + 1
            
        Theta.append(theta)
    
    taxa_aceit = na/N
    print(f"Lambda = {lamb}:")
    print("\tTaxa de aceitação:", taxa_aceit)
    
    # Descarta as amostras burnin para calcular a média a posteriori:
    Theta = np.array(Theta)
    Theta_post = Theta[burn:]
    print("\tMédia a posteriori:", Theta_post.mean())

    #%% Gráficos
    fig,ax = plt.subplots(ncols=2, figsize=(5,2), layout='constrained')
    ax[0].set_title('Traço da simulação')
    ax[0].plot(Theta)
    ax[0].axvline(100, color='C1', linestyle='--', label='burn-in')
    ax[0].set_xlabel('Passo')
    ax[0].set_ylabel(r'$\theta$')
    ax[0].legend()
    
    
    ax[1].set_title('Posteriori Simulada')
    ax[1].hist(Theta_post, bins=10, density=True, alpha=0.5, edgecolor='w')
    sns.kdeplot(Theta, bw_adjust=2, ax=ax[1], color='C0')
    ax[1].axvline(Theta.mean(), color='C1', linestyle='--', label='média')
    ax[1].set_xticks(np.arange(0,0.8,0.1))
    ax[1].set_xlim([0,0.7])
    ax[1].set_xlabel(r'$\theta$')
    ax[1].set_ylabel('Frequência normlizada')
    ax[1].legend()