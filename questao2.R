graphics.off()  # close all plots
rm(list=ls())	# clear the (global) environment variables
cat("\014") 	# clear the console (Ctrl+L)
set.seed(42)    # semente de simulação

# Print auxiliary function
printf <- function(...) {
    x = paste(sprintf(...),"\n")
    return(cat(x))
}

beta_pdf <- function(x, alpha, beta) {
    y <- dbeta(x, alpha, beta)
    return(y)
}

beta_cdf <- function(q, alpha, beta){
    y <- pbeta(q, alpha, beta)
    return(y)
}

norm_pdf <- function(x, mu, sigma){
    y <- dnorm(x, mean=mu, sd=sigma)
    return(y)
}

# Fuções auxiliares para cálculo dos intervalos

jeffrey <- function(alpha, beta){
    a <- qbeta(0.025, alpha, beta)
    b <-qbeta(0.975, alpha, beta)
    return(list(a=a, b=b))
}

jeffrey_modificado <- function(x, n, a1, b1){
    if (x == 0){
        a2 <- 0
        b2 <- b1
    } else{
        if (x == n) {
            a2 <- a1
            b2 <- 1
        } else{
            a2 <- a1
            b2 <- b1
        }
    }
    return(list(a=a2, b=b2))
}

wald <- function(x,n){
    theta_hat <- x/n
    sigma <- sqrt((theta_hat*(1-theta_hat))/n)
    a <- qnorm(0.025, theta_hat, sigma)
    b <- qnorm(0.975, theta_hat ,sigma)
    return(list(a=a, b=b))
}

wald_modificado <- function(x,n){
    theta_hat <- (x+2)/(n+4)
    sigma <- sqrt(theta_hat*(1-theta_hat)/(n+4))
    a <- qnorm(0.025, theta_hat, sigma)
    b <- qnorm(0.975, theta_hat, sigma)
    return(list(a=a, b=b))
}

# verifica se theta está em um dado intervalo
verifica_intervalo <- function(a,b,theta_hidden){
    if (theta_hidden >= a & theta_hidden <= b){
        coberto = TRUE
    } else{
        coberto = FALSE
    }
    return(coberto)
}

#####
# Simulação

N <- 10000 # número de experimentos
n <- 50    # tamanho de cada amostra

# Espaço amostral
Theta_hidden1 <- seq(0,    0.10, 0.001)
Theta_hidden2 <- seq(0.11, 0.89, 0.010)
Theta_hidden3 <- seq(0.90, 1,    0.001)
Theta_hidden <-c(Theta_hidden1, Theta_hidden2, Theta_hidden3)

# Vetores auxiliares
Prob_cobertura_jeffrey <- numeric(length(Theta_hidden))
Prob_cobertura_jeffrey_mod <- numeric(length(Theta_hidden))
Prob_cobertura_wald <- numeric(length(Theta_hidden))
Prob_cobertura_wald_mod <- numeric(length(Theta_hidden))


for (i in 1:length(Theta_hidden)){

    theta_hidden <- Theta_hidden[i]
    ns_jeffrey <- 0 # número de sucessos
    ns_jeffrey_mod <- 0
    ns_wald <- 0
    ns_wald_mod <- 0

    for (j in 1:N){

        # simula um valor para x
        x <- rbinom(1, size=n, prob=theta_hidden)

        # Hiperparãmetros da posteriori - theta|x ~ Beta(alpha, beta)
        alpha <- x+1/2
        beta <- n-x+1/2

        # Amostra theta
        theta <- rbeta(1, alpha, beta)

        # Calcula os intervalos de confiança

        # Jeffrey
        res_jeffrey <- jeffrey(alpha,beta)
        a_jeffrey <- res_jeffrey$a
        b_jeffrey <- res_jeffrey$b
        if (verifica_intervalo(a_jeffrey,b_jeffrey,theta_hidden)) {
            ns_jeffrey <- ns_jeffrey + 1
        }

        # Jeffrey Modificado
        res_jeffrey_mod <- jeffrey_modificado(x, n, a_jeffrey, b_jeffrey)
        a_jeffrey_mod <- res_jeffrey_mod$a
        b_jeffrey_mod <- res_jeffrey_mod$b
        if (verifica_intervalo(a_jeffrey_mod, b_jeffrey_mod, theta_hidden)) {
            ns_jeffrey_mod <- ns_jeffrey_mod + 1
        }

        # Wald
        res_wald <- wald(x, n)
        a_wald <- res_wald$a
        b_wald <- res_wald$b
        if (verifica_intervalo(a_wald, b_wald, theta_hidden)) {
            ns_wald <- ns_wald + 1
        }

        # Wald modificado
        res_wald_mod <- wald_modificado(x, n)
        a_wald_mod <- res_wald_mod$a
        b_wald_mod <- res_wald_mod$b
        if (verifica_intervalo(a_wald_mod, b_wald_mod, theta_hidden)) {
            ns_wald_mod <- ns_wald_mod + 1
        }

    }

    # Calcula a probabilidade de cobertura
    Prob_cobertura_jeffrey[i] <- ns_jeffrey/N
    Prob_cobertura_jeffrey_mod[i] <- ns_jeffrey_mod/N
    Prob_cobertura_wald[i] <- ns_wald/N
    Prob_cobertura_wald_mod[i] <- ns_wald_mod/N
}

#####
# Gráficos
plot(Theta_hidden, Prob_cobertura_jeffrey, t='l', lty=2, lwd=2,
     xlab=expression(theta), ylab="Prob. de cobertura")
lines(Theta_hidden, Prob_cobertura_jeffrey_mod, t='l', col='red')
lines(Theta_hidden, Prob_cobertura_wald, t='l', col='blue')
lines(Theta_hidden, Prob_cobertura_wald_mod, t='l', col='green')
abline(h=0.95, lt=2)
legend(0.6, 0.8, legend=c("Jeffrey", "Jeffrey Mod.", "Wald", "Wald Mod.", "95%"),
       col=c("black", "red", "blue", "green", "black"), lty=c(2,1,1,1,2), 
       lwd=c(2,1,1,1,1), cex=0.7)
grid()

plot(Theta_hidden1, Prob_cobertura_jeffrey[1:101], t='l', lty=2, lwd=2,
     xlab=expression(theta), ylab="Prob. de cobertura")
lines(Theta_hidden1, Prob_cobertura_jeffrey_mod[1:101], t='l', col='red')
lines(Theta_hidden1, Prob_cobertura_wald[1:101], t='l', col='blue')
lines(Theta_hidden1, Prob_cobertura_wald_mod[1:101], t='l', col='green')
abline(h=0.95, lt=2)
legend(0.08, 0.8, legend=c("Jeffrey", "Jeffrey Mod.", "Wald", "Wald Mod.", "95%"),
       col=c("black", "red", "blue", "green", "black"), lty=c(2,1,1,1,2), lwd=c(2,1,1,1,1), cex=0.7)
grid()

plot(Theta_hidden3, Prob_cobertura_jeffrey[181:281], t='l', lty=2, lwd=2, ylim=c(-0.01,1.01),
     xlab=expression(theta), ylab="Prob. de cobertura")
lines(Theta_hidden3, Prob_cobertura_jeffrey_mod[181:281], t='l', col='red')
lines(Theta_hidden3, Prob_cobertura_wald[181:281], t='l', col='blue')
lines(Theta_hidden3, Prob_cobertura_wald_mod[181:281], t='l', col='green')
abline(h=0.95, lt=2)
legend(0.9, 0.6, legend=c("Jeffrey", "Jeffrey Mod.", "Wald", "Wald Mod.", "95%"),
       col=c("black", "red", "blue", "green", "black"), lty=c(2,1,1,1,2), lwd=c(2,1,1,1,1), cex=0.7)
grid()
