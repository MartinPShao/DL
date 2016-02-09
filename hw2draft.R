house_train_input <- read.table(file = "./House_inputs_train.dat")
house_train_output <- read.table(file = "./House_output_train.dat")
house_test_input <- read.table(file = "./House_inputs_test.dat")
house_test_output <- read.table(file = "./House_output_test.dat")
house_train_input <- sapply(house_train_input, scale)
house_test_input <- sapply(house_test_input, scale)
sigm <- function(x) 1/(1 + exp(-x))
nn_bp_sgd <- function(input, output, gamma = 0.1, lambda = 0.1, 
                      maxiter = 100000, size = 10, tol = 1e-4, 
                      m = round(nrow(input)/3), rang = 0.01){
        x <- t(as.matrix(cbind(int = rep(1, nrow(input)), input)))
        y <- t(as.matrix(output))
        I <- nrow(x)
        J <- nrow(y)
        alpha <- matrix(data = runif(size*I, -rang, rang), ncol = nrow(x))
        beta <- matrix(data = runif(J*(size+1), -rang, rang), ncol = size+1)
        mse_past <- var(as.numeric(y))
        for(i in 1:maxiter){
                # Sampling
                ind <- sample(1:ncol(x), m)
                
                # Forward
                z <- sigm(rbind(int = rep(1, ncol(x[, ind])), alpha %*% x[, ind]))
                
                y_hat <- beta %*% z
                
                mse <- mean(as.numeric((y[, ind]-y_hat)^2))
                if (abs(mse_past - mse) <= tol) break
                mse_past <- mse
                # Backward
                beta_update <- -(y[, ind]-y_hat) %*% t(z)/length(y[, ind]) * gamma + 
                        lambda * gamma * c(0, beta[-1])/length(y[, ind])
                alpha_update <- -(do.call(rbind, replicate(size, y[, ind]-y_hat, 
                                                           simplify=FALSE)) * 
                                          z[-1, ] * (1 - z[-1, ]) * beta[, -1]) %*% 
                        t(x[, ind])/length(y[, ind]) * gamma + lambda * gamma *
                        cbind(numeric(size), alpha[, -1])/length(y[, ind])
                
                beta <- beta - beta_update
                alpha <- alpha - alpha_update
        }
        z <- sigm(rbind(int = rep(1, ncol(x)), alpha %*% x))
        y_hat <- beta %*% z
        mse <- mean(as.numeric((y-y_hat)^2))
        list(alpha, beta, mse, i)
        
}

predict_bp <- function(input, output, nn_bp_out){
        x <- t(as.matrix(cbind(int = rep(1, nrow(input)), input)))
        y <- t(as.matrix(output))
        alpha <- nn_bp_out[[1]]
        beta <- nn_bp_out[[2]]
        z <- sigm(rbind(int = rep(1, ncol(x)), alpha %*% x))
        y_hat <- beta %*% z
        mse <- mean(as.numeric((y-y_hat)^2))
        list(y_hat, mse)
        
}


nn_bp_1 <- nn_bp_sgd(input = house_train_input, output = house_train_output)
nnet_1 <- nnet(x = house_train_input, y = house_train_output, size = 10,
               linout = TRUE, rang = 0.01, decay = 0.1, maxit = 10000)
pred_bp_1 <- predict_bp(house_test_input, house_test_output, nn_bp_1)
pred_nnet_1 <- predict(nnet_1, house_test_input)
pred_nnet_1_mse <- mean((house_test_output - pred_nnet_1)^2)

# initial condition
ranges <- c(0.005, 0.01, 0.07, 0.1, 0.7)
dim_names <- list(paste("range=", ranges, sep = ""), 
                  c("BP(train)", "BP(test)", 
                    "nnet(train)", "nnet(test)"))
result_1_1 <- matrix(numeric(5*4), ncol = 4, dimnames = dim_names)
for (k in 1:5){
        nn_bp_1 <- nn_bp_sgd(input = house_train_input, 
                             output = house_train_output, 
                             rang = ranges[k])
        pred_bp_1 <- predict_bp(house_test_input, 
                                house_test_output, 
                                nn_bp_1)
        result_1_1[k, 1] <- nn_bp_1[[3]]
        result_1_1[k, 2] <- pred_bp_1[[2]]
        nnet_1 <- nnet(x = house_train_input, 
                       y = house_train_output, 
                       size = 10,
                       linout = TRUE, 
                       rang = ranges[k], 
                       decay = 0.1,
                       maxit = 10000)
        pred_nnet_1 <- predict(nnet_1, house_test_input)
        pred_nnet_1_mse <- mean((house_test_output - pred_nnet_1)^2)
        result_1_1[k, 3] <- mean(nnet_1$residuals^2)
        result_1_1[k, 4] <- pred_nnet_1_mse
}
var_train <- var(house_train_output[, 1])
var_test <- var(house_test_output[, 1])

# decay
decays <- c(0.001, 0.01, 0.1, 1, 10)
dim_names <- list(paste("decay=", decays, sep = ""), 
                  c("BP(train)", "BP(test)", 
                    "nnet(train)", "nnet(test)"))
result_1_2 <- matrix(numeric(5*4), ncol = 4, dimnames = dim_names)
for (k in 1:5){
        nn_bp_1 <- nn_bp_sgd(input = house_train_input, 
                             output = house_train_output, 
                             rang = 0.7, 
                             lambda = decays[k])
        pred_bp_1 <- predict_bp(house_test_input, 
                                house_test_output, 
                                nn_bp_1)
        result_1_2[k, 1] <- nn_bp_1[[3]]
        result_1_2[k, 2] <- pred_bp_1[[2]]
        nnet_1 <- nnet(x = house_train_input, 
                       y = house_train_output, 
                       size = 10,
                       linout = TRUE, 
                       rang = 0.7, 
                       decay = decays[k],
                       maxit = 10000)
        pred_nnet_1 <- predict(nnet_1, house_test_input)
        pred_nnet_1_mse <- mean((house_test_output - pred_nnet_1)^2)
        result_1_2[k, 3] <- mean(nnet_1$residuals^2)
        result_1_2[k, 4] <- pred_nnet_1_mse
}

# Learning
gammas <- seq(0.1, 1, 0.1)
dim_names <- list(c("nnet", paste("gamma=", gammas, sep = "")), 
                  c("train", "test"))
result_1_3 <- matrix(numeric(11*2), ncol = 2, dimnames = dim_names)
nnet_1 <- nnet(x = house_train_input, 
               y = house_train_output, 
               size = 10,
               linout = TRUE, 
               rang = 0.7, 
               decay = 10,
               maxit = 10000)
pred_nnet_1 <- predict(nnet_1, house_test_input)
pred_nnet_1_mse <- mean((house_test_output - pred_nnet_1)^2)
result_1_3[1, 1] <- mean(nnet_1$residuals^2)
result_1_3[1, 2] <- pred_nnet_1_mse
for (k in 1:10){
        nn_bp_1 <- nn_bp_sgd(input = house_train_input, 
                             output = house_train_output, 
                             rang = 0.7, 
                             lambda = 10, 
                             gamma = gammas[k])
        pred_bp_1 <- predict_bp(house_test_input, 
                                house_test_output, 
                                nn_bp_1)
        result_1_3[k+1, 1] <- nn_bp_1[[3]]
        result_1_3[k+1, 2] <- pred_bp_1[[2]]
}

# number of hidden variables
hiddens <- c(3, 5, 10, 20, 50)
dim_names <- list(paste("n_hidden=", hiddens, sep = ""), 
                  c("BP(train)", "BP(test)", 
                    "nnet(train)", "nnet(test)"))
result_1_4 <- matrix(numeric(5*4), ncol = 4, dimnames = dim_names)
for (k in 1:5){
        nn_bp_1 <- nn_bp_sgd(input = house_train_input, 
                             output = house_train_output, 
                             rang = 0.7, 
                             lambda = 10, 
                             gamma = 0.1, 
                             size = hiddens[k])
        pred_bp_1 <- predict_bp(house_test_input, 
                                house_test_output, 
                                nn_bp_1)
        result_1_4[k, 1] <- nn_bp_1[[3]]
        result_1_4[k, 2] <- pred_bp_1[[2]]
        nnet_1 <- nnet(x = house_train_input, 
                       y = house_train_output, 
                       size = hiddens[k],
                       linout = TRUE, 
                       rang = 0.7, 
                       decay = 10,
                       maxit = 10000)
        pred_nnet_1 <- predict(nnet_1, house_test_input)
        pred_nnet_1_mse <- mean((house_test_output - pred_nnet_1)^2)
        result_1_4[k, 3] <- mean(nnet_1$residuals^2)
        result_1_4[k, 4] <- pred_nnet_1_mse
}

# batch size
ms <- c(100, 200, 300, 400)
dim_names <- list(c("nnet", paste("m=", ms, sep = "")), 
                  c("train", "test"))
result_1_5 <- matrix(numeric(5*2), ncol = 2, dimnames = dim_names)
nnet_1 <- nnet(x = house_train_input, 
               y = house_train_output, 
               size = 10,
               linout = TRUE, 
               rang = 0.7, 
               decay = 10,
               maxit = 10000)
pred_nnet_1 <- predict(nnet_1, house_test_input)
pred_nnet_1_mse <- mean((house_test_output - pred_nnet_1)^2)
result_1_5[1, 1] <- mean(nnet_1$residuals^2)
result_1_5[1, 2] <- pred_nnet_1_mse
for (k in 1:4){
        nn_bp_1 <- nn_bp_sgd(input = house_train_input, 
                             output = house_train_output, 
                             rang = 0.7, 
                             lambda = 10, 
                             gamma = 0.1, 
                             m = ms[k])
        pred_bp_1 <- predict_bp(house_test_input, 
                                house_test_output, 
                                nn_bp_1)
        result_1_5[k+1, 1] <- nn_bp_1[[3]]
        result_1_5[k+1, 2] <- pred_bp_1[[2]]
}


wine_input <- read.table(file = "./Wine_input.dat")
wine_output <- read.table(file = "./Wine_output.dat")
ind <- sample(1:nrow(wine_input), round(nrow(wine_input)/5))
wine_train_input <- wine_input[-ind, ]
wine_train_output <- wine_output[-ind, ]
wine_test_input <- wine_input[ind, ]
wine_test_output <- wine_output[ind, ]
wine_train_input <- sapply(wine_train_input, scale)
wine_test_input <- sapply(wine_test_input, scale)

softmax <- function(y) exp(y)/sum(exp(y))
crossentropy <- function(y, yhat){
        -mean(apply(y * (log(yhat)), 1, sum))
}
nn_bp_sgd_2 <- function(input, output, gamma = 0.005, lambda = 1, 
                        maxiter = 100000, size = 10, tol = 1e-6, 
                        m = round(nrow(input)/3), rang = 0.01){
        x <- t(as.matrix(cbind(int = rep(1, nrow(input)), input)))
        y <- t(as.matrix(output))
        I <- nrow(x)
        J <- nrow(y)
        alpha <- matrix(data = runif(size*I, -rang, rang), ncol = nrow(x))
        beta <- matrix(data = runif(J*(size+1), -rang, rang), ncol = size+1)
        y_hat <- matrix(rep(1/3, nrow(y)*ncol(y)), 
                        nrow = nrow(y), ncol = ncol(y))
        ce_past <- crossentropy(t(y), t(y_hat))
        for(i in 1:maxiter){
                # Sampling
                ind <- sample(1:ncol(x), m)
                
                # Forward
                z <- sigm(rbind(int = rep(1, ncol(x[, ind])), alpha %*% x[, ind]))
                
                y_hat <- apply(beta %*% z, 2, softmax)
                
                ce <- crossentropy(t(y[, ind]), t(y_hat))
                print(ce)
                print(i)
                if (abs(ce_past - ce) <= tol) break
                ce_past <- ce
                # Backward
                beta_update <- -(y[, ind]-y_hat) %*% t(z)/length(y[, ind]) * gamma + 
                        lambda * gamma * c(0, beta[-1])/length(y[, ind])
                alpha_update <- -(t(beta[, -1]) %*% (y[, ind]-y_hat) * 
                        z[-1, ] * (1 - z[-1, ]))  %*% 
                        t(x[, ind])/length(y[, ind]) * gamma + lambda * gamma *
                        cbind(numeric(size), alpha[, -1])/length(y[, ind])
                
                beta <- beta - beta_update
                alpha <- alpha - alpha_update
        }
        z <- sigm(rbind(int = rep(1, ncol(x)), alpha %*% x))
        y_hat <- apply(beta %*% z, 2, softmax)
        ce <- crossentropy(t(y), t(y_hat))
        list(alpha, beta, y_hat, ce, i)
}

predict_bp_2 <- function(input, output, nn_bp_out){
        x <- t(as.matrix(cbind(int = rep(1, nrow(input)), input)))
        y <- t(as.matrix(output))
        alpha <- nn_bp_out[[1]]
        beta <- nn_bp_out[[2]]
        z <- sigm(rbind(int = rep(1, ncol(x)), alpha %*% x))
        y_hat <- apply(beta %*% z, 2, softmax)
        ce <- crossentropy(t(y), t(y_hat))
        list(y_hat, ce)
}


nn_bp_2 <- nn_bp_sgd_2(input = wine_train_input, output = wine_train_output)
nnet_2 <- nnet(x = wine_train_input, y = wine_train_output, size = 10,
               softmax = TRUE, rang = 0.01, decay = 10, maxit = 10000)
nnet_2_ce <- crossentropy(wine_train_output, nnet_2$fitted.values)
pred_bp_2 <- predict_bp_2(wine_test_input, wine_test_output, nn_bp_2)
pred_nnet_2 <- predict(nnet_2, wine_test_input)
pred_nnet_2_ce <- crossentropy(wine_test_output, pred_nnet_2)

# initial condition
ranges <- c(0.005, 0.01, 0.07, 0.1, 0.7)
dim_names <- list(paste("range=", ranges, sep = ""), 
                  c("BP(train)", "BP(test)", 
                    "nnet(train)", "nnet(test)"))
result_2_1 <- matrix(numeric(5*4), ncol = 4, dimnames = dim_names)
for (k in 1:5){
        nn_bp_2 <- nn_bp_sgd_2(input = wine_train_input, 
                               output = wine_train_output, 
                               rang = ranges[k])
        pred_bp_2 <- predict_bp_2(wine_test_input, 
                                  wine_test_output, 
                                  nn_bp_2)
        result_2_1[k, 1] <- nn_bp_2[[4]]
        result_2_1[k, 2] <- pred_bp_2[[2]]
        nnet_2 <- nnet(x = wine_train_input, 
                       y = wine_train_output, 
                       size = 10,
                       softmax = TRUE, 
                       rang = ranges[k], 
                       decay = 0.1,
                       maxit = 10000)
        pred_nnet_2 <- predict(nnet_2, wine_test_input)
        nnet_2_ce <- crossentropy(wine_train_output, nnet_2$fitted.values)
        pred_nnet_2_ce <- crossentropy(wine_test_output, pred_nnet_2)
        result_2_1[k, 3] <- nnet_2_ce
        result_2_1[k, 4] <- pred_nnet_2_ce
}

# decay
decays <- c(0.001, 0.01, 0.1, 1, 10)
dim_names <- list(paste("decay=", decays, sep = ""), 
                  c("BP(train)", "BP(test)", 
                    "nnet(train)", "nnet(test)"))
result_2_2 <- matrix(numeric(5*4), ncol = 4, dimnames = dim_names)
for (k in 1:5){
        nn_bp_2 <- nn_bp_sgd_2(input = wine_train_input, 
                               output = wine_train_output, 
                               rang = 0.7, 
                               lambda = decays[k])
        pred_bp_2 <- predict_bp_2(wine_test_input, 
                                  wine_test_output, 
                                  nn_bp_2)
        result_2_2[k, 1] <- nn_bp_2[[4]]
        result_2_2[k, 2] <- pred_bp_2[[2]]
        nnet_2 <- nnet(x = wine_train_input, 
                       y = wine_train_output, 
                       size = 10,
                       softmax = TRUE, 
                       rang = 0.7, 
                       decay = decays[k],
                       maxit = 10000)
        pred_nnet_2 <- predict(nnet_2, wine_test_input)
        nnet_2_ce <- crossentropy(wine_train_output, nnet_2$fitted.values)         
        pred_nnet_2_ce <- crossentropy(wine_test_output, pred_nnet_2)
        result_2_2[k, 3] <- nnet_2_ce
        result_2_2[k, 4] <- pred_nnet_2_ce
}

# Learning
gammas <- seq(0.005, 0.01, 0.001)
dim_names <- list(c("nnet", paste("gamma=", gammas, sep = "")), 
                  c("train", "test"))
result_2_3 <- matrix(numeric(7*2), ncol = 2, dimnames = dim_names)
nnet_2 <- nnet(x = wine_train_input, 
               y = wine_train_output, 
               size = 10,
               softmax = TRUE, 
               rang = 0.7, 
               decay = 1,
               maxit = 10000)
pred_nnet_2 <- predict(nnet_2, wine_test_input)
nnet_2_ce <- crossentropy(wine_train_output, nnet_2$fitted.values)         
pred_nnet_2_ce <- crossentropy(wine_test_output, pred_nnet_2)
result_2_3[1, 1] <- nnet_2_ce
result_2_3[1, 2] <- pred_nnet_2_ce
for (k in 1:6){
        nn_bp_2 <- nn_bp_sgd_2(input = wine_train_input, 
                               output = wine_train_output, 
                               rang = 0.7, 
                               lambda = 1, 
                               gamma = gammas[k])
        pred_bp_2 <- predict_bp_2(wine_test_input, 
                                  wine_test_output, 
                                  nn_bp_2)
        result_2_3[k+1, 1] <- nn_bp_2[[4]]
        result_2_3[k+1, 2] <- pred_bp_2[[2]]
}

# number of hidden variables
hiddens <- c(3, 5, 10, 20, 50)
dim_names <- list(paste("n_hidden=", hiddens, sep = ""), 
                  c("BP(train)", "BP(test)", 
                    "nnet(train)", "nnet(test)"))
result_2_4 <- matrix(numeric(5*4), ncol = 4, dimnames = dim_names)
for (k in 1:5){
        nn_bp_2 <- nn_bp_sgd_2(input = wine_train_input, 
                               output = wine_train_output, 
                               rang = 0.7, 
                               lambda = 1, 
                               gamma = 0.005, 
                               size = hiddens[k])
        pred_bp_2 <- predict_bp_2(wine_test_input, 
                                  wine_test_output, 
                                  nn_bp_2)
        result_2_4[k, 1] <- nn_bp_2[[4]]
        result_2_4[k, 2] <- pred_bp_2[[2]]
        nnet_2 <- nnet(x = wine_train_input, 
                       y = wine_train_output, 
                       size = hiddens[k],
                       softmax = TRUE, 
                       rang = 0.7, 
                       decay = 1,
                       maxit = 10000)
        pred_nnet_2 <- predict(nnet_2, wine_test_input)
        nnet_2_ce <- crossentropy(wine_train_output, nnet_2$fitted.values)         
        pred_nnet_2_ce <- crossentropy(wine_test_output, pred_nnet_2)
        result_2_4[k, 3] <- nnet_2_ce
        result_2_4[k, 4] <- pred_nnet_2_ce
}

# batch size
ms <- c(50, 100, 142)
dim_names <- list(c("nnet", paste("m=", ms, sep = "")), 
                  c("train", "test"))
result_2_5 <- matrix(numeric(4*2), ncol = 2, dimnames = dim_names)
nnet_2 <- nnet(x = wine_train_input, 
               y = wine_train_output, 
               size = 50,
               softmax = TRUE, 
               rang = 0.7, 
               decay = 1,
               maxit = 10000)
pred_nnet_2 <- predict(nnet_2, wine_test_input)
nnet_2_ce <- crossentropy(wine_train_output, nnet_2$fitted.values)         
pred_nnet_2_ce <- crossentropy(wine_test_output, pred_nnet_2)
result_2_5[1, 1] <- nnet_2_ce
result_2_5[1, 2] <- pred_nnet_2_ce
for (k in 1:3){
        nn_bp_2 <- nn_bp_sgd_2(input = wine_train_input, 
                               output = wine_train_output, 
                               rang = 0.7, 
                               lambda = 1, 
                               gamma = 0.1, 
                               m = ms[k], 
                               size = 50)
        pred_bp_2 <- predict_bp_2(wine_test_input, 
                                  wine_test_output, 
                                  nn_bp_2)
        result_2_5[k+1, 1] <- nn_bp_2[[4]]
        result_2_5[k+1, 2] <- pred_bp_2[[2]]
}
y_hat <- matrix(rep(1/3, nrow(wine_test_output)*ncol(wine_test_output)), 
                nrow = nrow(wine_test_output), ncol = ncol(wine_test_output))
ce_test_base <- crossentropy(wine_test_output, y_hat)
y_hat <- matrix(rep(1/3, nrow(wine_train_output)*ncol(wine_train_output)), 
                nrow = nrow(wine_train_output), ncol = ncol(wine_train_output))
ce_train_base <- crossentropy(wine_train_output, y_hat)
save(list = ls(), file = "./hw2output.RData")

