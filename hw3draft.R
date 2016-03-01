setwd("~/Documents/git/DL")

# 1
# minst0_test <- read.table(file = "./minst0_test.dat")
# minst0_train <- read.table(file = "./minst0_train.dat")
# minst9_test <- read.table(file = "./minst9_test.dat")
# minst9_train <- read.table(file = "./minst9_train.dat")
# save(list = c("minst0_test", "minst0_train", "minst9_test", "minst9_train"), 
#      file = "./hw3q1_data.RData")


load(file = "./hw3q1_data.RData")
library(EBImage)
draw <- function(mat){
        image(t(mat)[,ncol(mat):1], axes = FALSE, col = grey(seq(0, 1, length = 256)))
}
par(mfrow = c(3, 3), oma = c(0, 0, 0, 0), mar = c(1, 1, 1, 1))


m <- matrix(data = unlist(minst9_train[1, ]), nrow = 28, byrow = TRUE)
draw(m)
#a, rotate, clockwise, 90 degree
ya <- matrix(c(0, 1, 0, 0, 0, 0, 0, 0, 0), nrow = 3)
draw(filter2(m, ya))
yb <- matrix(c(0, 0, 0, 0, 0, 0, 0, 1, 0), nrow = 3)
draw(filter2(m, yb))
yc <- matrix(c(0, 0, 0, 1, 0, 0, 0, 0, 0), nrow = 3)
draw(filter2(m, yc))
yd <- matrix(c(0, 0, 0, 0, 0, 1, 0, 0, 0), nrow = 3)
draw(filter2(m, yd))
ye1 <- matrix(c(-1, 0, 1, -2, 0, 2, -1, 0, 1), nrow = 3)
draw(filter2(m, ye1))
ye2 <- matrix(c(1, 2, 1, 0, 0, 0, -1, -2, -1), nrow = 3)
draw(filter2(m, ye2))
ye3 <- matrix(c(0, -1, 0, -1, 4, -1, 0, -1, 0), nrow = 3)
draw(filter2(m, ye3))
ye4 <- matrix(c(-1, -1, -1, -1, 8, -1, -1, -1, -1), nrow = 3)
draw(filter2(m, ye4))

# b
par(mfrow = c(2, 3))
yb_3 <- matrix(rep(1/3^2, 3^2), nrow = 3)
yb_5 <- matrix(rep(1/5^2, 5^2), nrow = 5)
yb_7 <- matrix(rep(1/7^2, 7^2), nrow = 7)
draw(filter2(m, yb_3))
draw(filter2(m, yb_5))
draw(filter2(m, yb_7))
ys_3 <- matrix(numeric(3^2), nrow = 3)
ys_3[(3+1)/2, (3+1)/2] <- 2
ys_3 <- ys_3 - yb_3
ys_5 <- matrix(numeric(5^2), nrow = 5)
ys_5[(5+1)/2, (5+1)/2] <- 2
ys_5 <- ys_5 - yb_5
ys_7 <- matrix(numeric(7^2), nrow = 7)
ys_7[(7+1)/2, (7+1)/2] <- 2
ys_7 <- ys_7 - yb_7
draw(filter2(m, ys_3))
draw(filter2(m, ys_5))
draw(filter2(m, ys_7))
# yu_3 <- matrix(runif(3^2), nrow = 3)
# yu_5 <- matrix(runif(5^2), nrow = 5)
# yu_7 <- matrix(runif(7^2), nrow = 7)
# image(filter2(m, yu_3), axes = FALSE, col = grey(seq(0, 1, length = 256)))
# image(filter2(m, yu_5), axes = FALSE, col = grey(seq(0, 1, length = 256)))
# image(filter2(m, yu_7), axes = FALSE, col = grey(seq(0, 1, length = 256)))

# c
pooling <- function(x, pool, method){
        m <- nrow(x)/pool
        n <- ncol(x)/pool
        z <- matrix(nrow = m, ncol = n)
        func <- ifelse(method == "mean", mean, max)
        for (i in 1:m){
                for (j in 1:n){
                        z[i,j] <- func(x[(pool*(i-1)+1):(pool*i), 
                                         (pool*(j-1)+1):(pool*j)])
                }
        }
        z
}

par(mfrow = c(2, 3))
draw(m)
draw(pooling(m,4,"mean"))
draw(pooling(m,4,"max"))
draw(m)
draw(pooling(m,7,"mean"))
draw(pooling(m,7,"max"))


## d
cov_laplacian <- filter2(m, ye3)
detect_ReLU <- ifelse(cov_laplacian>0, cov_laplacian, 0)
pool_max <- pooling(detect_ReLU,4,"max")
par(mfrow = c(2, 2))
draw(m)
draw(cov_laplacian)
draw(detect_ReLU)
draw(pool_max)


## e
train_data <- rbind(cbind(minst9_train, label = 1)[1:100, ], 
                    cbind(minst0_train, label = 0)[1:100, ])
sigmoid <- function(x) 1/(1+exp(-x))
forward_prop <- function(pars){
        kern_1 <- matrix(pars[1:9], nrow = 3, ncol = 3)
        kern_2 <- matrix(pars[10:18], nrow = 3, ncol = 3)
        kern_3 <- matrix(pars[19:27], nrow = 3, ncol = 3)
        kern_4 <- matrix(pars[28:36], nrow = 3, ncol = 3)
        kern_5 <- matrix(pars[37:45], nrow = 3, ncol = 3)
        alpha_0 <- matrix(pars[46:55], ncol = 10)
        alpha_1 <- matrix(pars[56:255], ncol = 10)
        beta_0 <- matrix(pars[256:256], ncol = 1)
        beta_1 <- matrix(pars[257:266], ncol = 1)
        outs <- numeric(nrow(train_data))
        for (i in 1:nrow(train_data)){
                temp_m <- matrix(data = unlist(train_data[i, -785]), 
                                 nrow = 28, byrow = TRUE)
                cov_laplacian_1 <- filter2(temp_m, kern_1)
                detect_ReLU_1 <- ifelse(cov_laplacian_1>0, cov_laplacian_1, 0)
                pool_max_1 <- pooling(detect_ReLU_1,14,"max")
                cov_laplacian_2 <- filter2(temp_m, kern_2)
                detect_ReLU_2 <- ifelse(cov_laplacian_2>0, cov_laplacian_2, 0)
                pool_max_2 <- pooling(detect_ReLU_2,14,"max")
                cov_laplacian_3 <- filter2(temp_m, kern_3)
                detect_ReLU_3 <- ifelse(cov_laplacian_3>0, cov_laplacian_3, 0)
                pool_max_3 <- pooling(detect_ReLU_3,14,"max")
                cov_laplacian_4 <- filter2(temp_m, kern_4)
                detect_ReLU_4 <- ifelse(cov_laplacian_4>0, cov_laplacian_4, 0)
                pool_max_4 <- pooling(detect_ReLU_4,14,"max")
                cov_laplacian_5 <- filter2(temp_m, kern_5)
                detect_ReLU_5 <- ifelse(cov_laplacian_5>0, cov_laplacian_5, 0)
                pool_max_5 <- pooling(detect_ReLU_5,14,"max")
                input <- t(as.numeric(c(pool_max_1, pool_max_2, pool_max_3, pool_max_4, pool_max_5)))
                output <- (input %*% alpha_1 + alpha_0) %*% beta_1 + beta_0
                outs[i] <- sigmoid(output)
        }
        outs
}

cost_function <- function(y, yhat){
        -mean(y*log(yhat)+(1-y)*log(1-yhat))
}


# start_time <- Sys.time()
# forward_prop(pars)
# dur <- Sys.time() - start_time


INITIAL_PARTICLE <- function(){
        cur_loc <- runif(266, -0.1, 0.1)
        cur_velocity <- runif(266, -0.1, 0.1)
        cur_pred <- forward_prop(cur_loc)
        cur_cost <- cost_function(train_data$label, cur_pred)
        best_loc <- cur_loc
        best_cost <- cur_cost
        return(list(cur_loc = cur_loc, 
                    cur_velocity = cur_velocity, 
                    cur_cost = cur_cost,
                    best_loc = best_loc, 
                    best_cost = best_cost))
}
# start_time <- Sys.time()
# p1 <- INITIAL_PARTICLE()
# dur <- Sys.time() - start_time
# Time difference of 2.691472 secs

INITIAL_SWARM <- function(n){
        swarm <- list()
        for (i in 1:n){
                swarm[[i]] <- INITIAL_PARTICLE()
        }
        return(swarm)
}
# start_time <- Sys.time()
# swarm <- INITIAL_SWARM(5)
# dur <- Sys.time() - start_time
# Time difference of 13.50203 secs

GET_GLOBAL_BEST <- function(swarm){
        g_best <- swarm[[1]]
        for (i in 1:length(swarm)){
                if (swarm[[i]]$cur_cost < g_best$cur_cost){
                        g_best <- swarm[[i]]
                }
        }
        return(g_best)
}

UPDATE_VELOCITY <- function(particle, g_best, w = 0.729, c1 = 1.49445, c2 = 1.49445, max_v){
        v1 <- c1 * runif(266) * (particle$best_loc - particle$cur_loc)
        v2 <- c2 * runif(266) * (g_best$best_loc - particle$cur_loc)
        particle$cur_velocity <- ifelse(w * particle$cur_velocity + v1 + v2 > max_v, 
                                        max_v, 
                                        ifelse(w * particle$cur_velocity + v1 + v2 < -max_v, 
                                               -max_v, 
                                               w * particle$cur_velocity + v1 + v2))
        return(particle)
}

UPDATE_LOCATION <- function(particle, bond){
        particle$cur_loc <- ifelse(particle$cur_loc + particle$cur_velocity > bond[2], 
                                   bond[2], 
                                   ifelse(particle$cur_loc + particle$cur_velocity < bond[1]), 
                                   bond[1], 
                                   particle$cur_loc + particle$cur_velocity)
        cur_pred <- forward_prop(particle$cur_loc)
        particle$cur_cost <- cost_function(train_data$label, cur_pred)
        if (particle$cur_cost < particle$best_cost){
                particle$best_cost <- particle$cur_cost
                particle$best_loc <- particle$cur_loc
        }
        return(particle)
}



SEARCH <- function(iter = 1000, size = 20, w = 0.729, 
                   c1 = 1.49445, c2 = 1.49445, max_v = 2, bond = c(-10, 10)){
        swarm <- INITIAL_SWARM(size)
        g_best <- GET_GLOBAL_BEST(swarm)
        for (i in 1:iter){
                for (j in length(swarm)){
                        swarm[[j]] <- UPDATE_VELOCITY(swarm[[j]], 
                                                      g_best, 
                                                      w, 
                                                      c1, 
                                                      c2, 
                                                      max_v)
                        swarm[[j]] <- UPDATE_LOCATION(swarm[[j]], bond)
                }
                g_best <- GET_GLOBAL_BEST(swarm)
                return(g_best)
        }
}

# start_time <- Sys.time()
# swarm <- SEARCH(10)
# dur <- Sys.time() - start_time
# Time difference of 13.50203 secs
















## 2



