myst_im <- as.matrix(read.table(file = "./myst_im.dat"))
draw <- function(mat, main = ""){
        image(t(mat)[,ncol(mat):1], axes = FALSE, col = grey(seq(0, 1, length = 256)), main=main)
}
draw(myst_im)
library(MCMCpack)
q <- 3
r <- 0.1
s <- 1
B <- 1
epsilon <- 0.001
n <- 1
M <- 1
rows <- nrow(myst_im)
cols <- ncol(myst_im)
y <- as.numeric(myst_im)
NEIGHBORS <- function(x, i, rows){
        neighbor <- c()
        if (i+1<=length(x)) neighbor <- c(neighbor, x[i+1])
        if (i+rows<=length(x)) neighbor <- c(neighbor, x[i+rows])
        if (i-1>0) neighbor <- c(neighbor, x[i-1])
        if (i-rows>0) neighbor <- c(neighbor, x[i-rows])
        return(neighbor)
}

SUMMARY_STAT <- function(x, rows){
        s <- 0
        for (i in 1:length(x))
                s <- s + sum(x[i], NEIGHBORS(x, i, rows))
        return(s)
}
SAMPLE_X <- function(x, y, beta, sigma, rows){
        x_prime <- 1-x
        for (i in 1:length(x)){
                d <- beta*sum(NEIGHBORS(x, i, rows)==x[i])+(-1/(2*sigma)*(y[i]-x[i])^2)
                d_prime <- beta*sum(NEIGHBORS(x, i, rows)==x_prime[i])+(-1/(2*sigma)*(y[i]-x_prime[i])^2)
                p <- exp(min(c(d_prime-d), 0))
                U <- runif(1)
                if (U<p)
                        x[i] <- x_prime[i]
        }
        return(x)
}
SAMPLE_SIGMA <- function(x, y, q, r){
        q <- q + length(y)/2
        r <- 1/(1/r + 0.5 * sum((y - x)^2))
        sigma <- rinvgamma(1, shape = q, scale = r)
        return(sigma)
}
SAMPLE_BETA <- function(beta, x, epsilon, s, B, M, rows){
        repeat{
                beta_star <- rnorm(1, beta, s)
                if (beta_star>0 & beta_star<B) break
        }
        w <- x
        for (j in 1:M){
                w_prime <- 1-w
                for (i in 1:length(x)){
                        d <- beta*sum(NEIGHBORS(w, i, rows)==w[i])
                        d_prime <- beta*sum(NEIGHBORS(w, i, rows)==w_prime[i])
                        p <- exp(min(c(d_prime-d), 0))
                        U <- runif(1)
                        if (U<p)
                                w[i] <- w_prime[i]
                }
                
        }
        if (abs(SUMMARY_STAT(x, rows)-SUMMARY_STAT(w, rows))<epsilon*SUMMARY_STAT(x, rows)){
                ratio <- (dnorm(beta_star))/(dnorm(beta))*
                        (dnorm(beta, beta_star, s))/(dnorm(beta_star, beta, s))
                if (runif(1)<ratio)
                        beta <- beta_star
        }
        return(beta)
}

xs_q2c <- list()
betas <- numeric(n)
sigmas <- numeric(n)
dur_q2c <- Sys.time()
x <- sample(c(0, 1), rows*cols, replace = TRUE)
repeat{
        beta <- rnorm(1, 0, s)
        if (beta>0 & beta<B) break
}
sigma <- rinvgamma(1, shape = q, scale = r)
for (t in 1:n){
        x <- SAMPLE_X(x, y, beta, sigma, rows)
        sigma <- SAMPLE_SIGMA(x, y, q, r)
        beta <- SAMPLE_BETA(beta, x, epsilon, s, B, M, rows)
}
dur_q2c <- Sys.time() - dur_q2c

