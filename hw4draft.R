setwd("~/Documents/git/DL")
# rm(list = ls())
# minst9bw_train <- read.table(file = "./minst9bw_train.dat")
# minst0bw_train <- read.table(file = "./minst0bw_train.dat")
# minstbw_train <- rbind(minst9bw_train, minst0bw_train)
# save(list = ls(), file = "./hw4datatsets.RData")
load(file = "./hw4datatsets.RData")
# a. 

batch_size <- c(100, 500)
m <- 784
n <- c(10, 50, 100)
eta <- c(0.001, 0.01, 0.1)
niter <- c(100, 150)
params <- expand.grid(batch_size, n, eta, niter)
names(params) <- c("batch_size", "n", "eta", "niter")
K <- nrow(params)
results <- list()
dur_time <- character(K)
w0 <- matrix(runif(m*n, -0.1, 0.1), nrow = n, ncol = m)
b0 <- matrix(runif(m*1, -0.1, 0.1), nrow = m, ncol = 1)
c0 <- matrix(runif(n*1, -0.1, 0.1), nrow = n, ncol = 1)
for (k in 1:K){
        w <- w0
        b <- b0
        c <- c0
        starting <- Sys.time()
        
        for (t in 1:params[k, "niter"]){
                sample.ind <- sample(1:nrow(minstbw_train), 
                                     params[k, "batch_size"])
                err <- numeric(params[k, "niter"])
                for (i in 1:params[k, "batch_size"]){
                        v <- unlist(minstbw_train[sample.ind[i], ])
                        h <- 1/(1 + exp(-(w %*% v + c)))
                        hs <- ifelse(h > 0.5, 1, 0)
                        vr <- 1/(1 + exp(-(t(w) %*% hs + b)))
                        hr <- 1/(1 + exp(-(w %*% vr + c)))
                        dw <- eta * (h%*%t(v) - hr%*%t(vr)) / params[k, "batch_size"]
                        db <- eta * (v - vr) / params[k, "batch_size"]
                        dc <- eta * (h - hr) / params[k, "batch_size"]
                        w <- w + dw
                        b <- b + db
                        c <- c + dc
                        err[t] <- err[t] + sum((v - vr)^2) / params[k, "batch_size"]
                }
        }
        dur_time[k] <- paste(Sys.time() - starting, 
                             attr(Sys.time() - starting, "units"))
        results[[k]] <- err
}
