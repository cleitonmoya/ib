graphics.off()  # close all plots
rm(list=ls())	# clear the (global) environment variables
cat("\014") 	# clear the console (Ctrl+L)

# Print auxiliary function
printf <- function(...) {
    x = paste(sprintf(...),"\n")
    return(cat(x))
}

n <- 50
x <- 25

# Parãmetros da posteriori (Beta)
alpha <- x+1/2
beta <- n-x+1/2
if (beta <= 0){
    beta <- 0.001
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

# Intervalo de confiança 95% (Jeffrey)
theta_hat <- alpha/(alpha+beta)
a1 <- qbeta(0.025, alpha, beta)
b1 <-qbeta(0.975, alpha, beta)
printf("Jeffrey: a1: %.3f", a1)
printf("Jeffrey: b1: %.3f\n", b1)

# Jeffrey modificado
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
    return(list(a2=a2, b2=b2))
}

ic_jeffrey_mod <- jeffrey_modificado(x, n, a1, b1)
a2 <- ic_jeffrey_mod$a2
b2 <- ic_jeffrey_mod$b2
printf("Jeffrey modificado: a2: %.3f", a2)
printf("Jeffrey modificado: b2: %.3f\n", b2)

# Intervalo de confiança de Wald para grandes amostras
theta_hat1 <- x/n
mu1 <- theta_hat1
sigma1 <- sqrt((theta_hat1*(1-theta_hat1))/n)
a_wald1 <- qnorm(0.025, mu1, sigma1)
b_wald1 <- qnorm(0.975, mu1 ,sigma1)
printf("IC Wald: a1: %.3f", a_wald1)
printf("IC Wald: b1: %.3f\n", b_wald1)

# Intervalo de confiança de Wald modificado
theta_hat2 <- (x+2)/(n+4)
mu2 <- theta_hat2
sigma2 <- sqrt((theta_hat2*(1-theta_hat2))/(n+4))
a_wald2 <- qnorm(0.025, mu2, sigma2)
b_wald2 <- qnorm(0.975, mu2 ,sigma2)
printf("IC Wald Modificado: a2: %.3f", a_wald2)
printf("IC Wald Modificado: b2: %.3f\n", b_wald2)

# Gráficos
theta <- seq(0, 1, 0.01)
y_beta_pdf <- sapply(theta, beta_pdf, alpha=alpha, beta=beta)
plot(theta, y_beta_pdf, t='l')
abline(v=a1, col='red')
abline(v=b1, col='red')

y_wald1 <- sapply(theta, norm_pdf, mu=mu1, sigma=sigma1)
y_wald2 <- sapply(theta, norm_pdf, mu=mu2, sigma=sigma2)
plot(theta, y_wald1, t='l')
lines(theta, y_wald2, col='blue')

#####
y_beta_pdf2 <- sapply(theta, beta_pdf, alpha=2, beta=2)
plot(theta,y_beta_pdf2,t='l', xlab=expression(theta), ylab="densidade")
