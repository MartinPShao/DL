setwd("~/Documents/git/DL")
load("./hw5data.RData")
sigm <- function(a){
        1/(1+exp(-a))
}

FORWARD_PROPAGATION <- function(x, W1, b1, W2, b2, m, n, K){
        z2 <- W1 %*% x + matrix(rep(b1, m), nr = n)
        h <- sigm(z2)
        z3 <- W2 %*% z2 + matrix(rep(b2, m), nr = K)
        y <- z3
        return(list(y = y, h = h))
}
BACKWARD_PROPAGATION <- function(fd, x, W2, beta, rho){
        delta3 <- -(x - fd$y)
        KL <- 0
        if (beta != 0){
                rho_hat <- apply(fd$h, 1, mean)
                KL <- -rho/(rho_hat + 1e-9) + (1 - rho)/(1 - rho_hat + 1e-9)
        }
        delta2 <- (t(W2) %*% delta3 + beta * KL) * fd$h * (1 - fd$h)
        return(list(delta2 = delta2, delta3 = delta3))
}
UPDATE_W1 <- function(delta, x, W1, alpha, lambda, m){
        return(W1 - alpha * (delta$delta2 %*% t(x)/m + lambda * W1))
}
UPDATE_b1 <- function(delta, b1, alpha){
        return(b1 - alpha * apply(delta$delta2, 1, mean))
}
UPDATE_W2 <- function(delta, fd, W2, alpha, lambda, m){
        return(W2 - alpha * (delta$delta3 %*% t(fd$h)/m + lambda * W2))
}
UPDATE_b2 <- function(delta, b2, alpha){
        return(b2 - alpha * apply(delta$delta3, 1, mean))
}
single_autoencoder <- function(input, 
                               alpha = 0.01, # learning rate
                               beta = 0, # KL sparseness penalty
                               lambda = 0.01, # regularization penalty
                               maxiter = 10, # maximum iteration times
                               n = 10, # number of hidden layers
                               rho = 0.05, # sparsity parameter
                               tol = 1e-4, # tolerance
                               m = round(nrow(input)/10) # mini-batch size
){
        # Initialization
        N <- nrow(input)
        K <- ncol(input)
        W1 <- matrix(runif(n*K, -0.1, 0.1), nr = n)
        b1 <- matrix(runif(n, -0.1, 0.1))
        W2 <- matrix(runif(K*n, -0.1, 0.1), nr = K)
        b2 <- matrix(runif(K, -0.1, 0.1))
        input <- t(input)
        output <- input
        h <- matrix(0, nr = n, nc = m)
        ybar <- apply(output, 1, mean)
        Q_past <- sum((output - matrix(rep(ybar, N), ncol = N))^2)/N/2
        Q <- c()
        for (i in 1:maxiter){
                sample_ind <- sample(1:N, m)
                x <- input[, sample_ind]
                y <- output[, sample_ind]
                fp <- FORWARD_PROPAGATION(x, W1, b1, W2, b2, m, n, K)
                bp <- BACKWARD_PROPAGATION(fp, x, W2, beta, rho)
                W1 <- UPDATE_W1(bp, x, W1, alpha, lambda, m)
                b1 <- UPDATE_b1(bp, b1, alpha)
                W2 <- UPDATE_W2(bp, fp, W2, alpha, lambda, m)
                b2 <- UPDATE_b2(bp, b2, alpha)
                Q_cur <- sum((y - fp$y)^2)/m/2
                Q <- c(Q, Q_cur)
                cat("Iteration: ",i,", Q=",Q[i],"\n",sep="")
                # if (abs(Q_cur - Q_past) <= tol) break
        }
        return(list(W1 = W1, b1 = b1, W2 = W2, b2 = b2, err = Q, iter = i))
}
start.time <- Sys.time()
tt <- single_autoencoder(minst0_train[1:1000, ], m = 1, maxiter = 10)
dur <- Sys.time() - start.time