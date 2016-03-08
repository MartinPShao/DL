
library(MCMCpack)
library(MASS)
x <- iris3[, 1, 1]
y <- iris3[, 2, 1]
q <- 3
r <- 0.1
Sigma <- diag(c(1, 1))
epsilon <- 0.1
n <- 10000
sims_abc <- matrix(nrow = n, ncol = 3)
k=0
for (i in 1:n){
        repeat{
                beta <- mvrnorm(1, mu = c(0, 0), Sigma)
                sigma <- rinvgamma(1, q, r)
                y_hat <- beta[1] + beta[2] * x + sigma
                print(k)
                if (mean((y-y_hat)^2) <= 0.05*mean(y)) break
        }
        sims_abc[i, ] <- c(beta, sigma)
}

# Gibbs Sampler
mu0 <- 0
tau0 <- 1
a <- 3
b <- 0.1
sims_gibbs <- matrix(nrow = n, ncol = 3)
beta0 <- rnorm(1, mu0, 1/tau0)
beta1 <- rnorm(1, mu0, 1/tau0)
tau <- rgamma(1, shape = a, rate = b)
for (i in 1:n){
        mu0_n <- (tau0*mu0+tau*sum(y-beta1*x))/(tau0+50*tau)
        tau0_n <- (tau0+50*tau)
        beta0 <- rnorm(1, mean = mu0_n, sd = 1/tau0_n)
        mu1_n <- (tau0*mu0+tau*sum(x*(y-beta0)))/(tau0+tau*sum(x^2))
        tau1_n <- (tau0+tau*sum(x^2))
        beta1 <- rnorm(1, mean = mu1_n, sd = 1/tau1_n)
        a_n <- a + 50/2
        b_n <- b + sum((y-beta0-beta1*x)^2)
        tau <- rgamma(1, shape = a, rate = b)
        sims_gibbs[i, ] <- c(beta0, beta1, 1/tau)
}
