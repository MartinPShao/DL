house_train_input <- read.table(file = "./House_inputs_train.dat")
house_train_output <- read.table(file = "./House_output_train.dat")
house_test_input <- read.table(file = "./House_inputs_test.dat")
house_test_output <- read.table(file = "./House_output_test.dat")
sigm <- function(x) 1/(1 + exp(-x))
# nn_bp_sgd <- function(input, output, gamma = 0.1, lambda = 0.1, 
#                       maxiter = 100000, size = 10, tol = 1e-4, 
#                       m = round(nrow(input)/3)){
#         x <- t(as.matrix(cbind(int = rep(1, nrow(input)), input)))
#         y <- t(as.matrix(output))
#         I <- nrow(x)
#         J <- nrow(y)
#         alpha <- matrix(data = runif(size*I, -0.01, 0.01), ncol = nrow(x))
#         beta <- matrix(data = runif(J*(size+1), -0.01, 0.01), ncol = nrow(z))
#         mse_past <- var(as.numeric(y))
#         for(i in 1:maxiter){
#                 # Sampling
#                 ind <- sample(1:ncol(x), m)
#                 
#                 # Forward
#                 z <- sigm(rbind(int = rep(1, ncol(x[, ind])), alpha %*% x[, ind]))
#                 
#                 y_hat <- beta %*% z
#                 
#                 mse <- mean(as.numeric((y[, ind]-y_hat)^2))
#                 if (abs(mse_past - mse) <= tol) break
#                 mse_past <- mse
#                 # Backward
#                 beta_update <- -(y[, ind]-y_hat) %*% t(z)/length(y[, ind]) * gamma + 
#                         lambda * gamma * c(0, beta[-1])/length(y[, ind])
#                 alpha_update <- -(do.call(rbind, replicate(size, y[, ind]-y_hat, 
#                                                            simplify=FALSE)) * 
#                                           z[-1, ] * (1 - z[-1, ]) * beta[, -1]) %*% 
#                         t(x[, ind])/length(y[, ind]) * gamma + lambda * gamma *
#                         cbind(numeric(size), alpha[, -1])/length(y[, ind])
#                 
#                 beta <- beta - beta_update
#                 alpha <- alpha - alpha_update
#         }
#         z <- sigm(rbind(int = rep(1, ncol(x)), alpha %*% x))
#         y_hat <- beta %*% z
#         mse <- mean(as.numeric((y-y_hat)^2))
#         list(alpha, beta, mse, i)
#         
# }
# 
# predict_bp <- function(input, output, nn_bp_out){
#         x <- t(as.matrix(cbind(int = rep(1, nrow(input)), input)))
#         y <- t(as.matrix(output))
#         alpha <- nn_bp_out[[1]]
#         beta <- nn_bp_out[[2]]
#         z <- sigm(rbind(int = rep(1, ncol(x)), alpha %*% x))
#         y_hat <- beta %*% z
#         mse <- mean(as.numeric((y-y_hat)^2))
#         list(y_hat, mse)
#         
# }
# 
# 
# nn_bp_1 <- nn_bp_sgd(input = house_train_input, output = house_train_output, 
#                      m = 300)
# nnet_1 <- nnet(x = house_train_input, y = house_train_output, size = 10,
#                linout = TRUE, rang = 0.01, decay = 0.1)
# pred_bp_1 <- predict_bp(house_test_input, house_test_output, nn_bp_1)
# pred_nnet_1 <- predict(nnet_1, house_test_input)
# pred_nnet_1_mse <- mean((house_test_output - pred_nnet_1)^2)
# 


wine_input <- read.table(file = "./Wine_input.dat")
wine_output <- read.table(file = "./Wine_output.dat")
ind <- sample(1:nrow(wine_input), round(nrow(wine_input)/5))
wine_train_input <- wine_input[-ind, ]
wine_train_output <- wine_output[-ind, ]
wine_test_input <- wine_input[ind, ]
wine_test_output <- wine_output[ind, ]

softmax <- function(y) exp(y)/sum(exp(y))
crossentropy <- function(y, yhat){
        -mean(apply(y * (log(yhat)), 1, sum))
}
nn_bp_sgd_2 <- function(input, output, gamma = 0.005, lambda = 1, 
                        maxiter = 100000, size = 10, tol = 1e-6, 
                        m = round(nrow(input)/3)){
        x <- t(as.matrix(cbind(int = rep(1, nrow(input)), input)))
        y <- t(as.matrix(output))
        I <- nrow(x)
        J <- nrow(y)
        alpha <- matrix(data = runif(size*I, -0.01, 0.01), ncol = nrow(x))
        beta <- matrix(data = runif(J*(size+1), -0.01, 0.01), ncol = nrow(z))
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
               softmax = TRUE, rang = 0.01, decay = 10)
nnet_2_ce <- crossentropy(wine_train_output, nnet_2$fitted.values)
pred_bp_2 <- predict_bp_2(wine_test_input, wine_test_output, nn_bp_2)
pred_nnet_2 <- predict(nnet_2, wine_test_input)
pred_nnet_2_ce <- crossentropy(wine_train_output, pred_nnet_2)




