rm(list = ls())

## 1
exprs <- expression((theta0+theta1*x)/(1+theta2*exp(theta3*x)))
theta0 <- 4.0
theta1 <- -1.5
theta2 <- 0.6
theta3 <- 0.4
x <- seq(0, 20, length.out = 1000)
epsilon <- rnorm(1000, 0, sqrt(0.1))
y <- eval(exprs) + epsilon


alpha <- 0.5 # learning parameter
m <- 50 # batch size
params <- list(theta0 = 4, theta1 = -1.5, theta2 = 0.6, theta3 = 0.4)
obj <- function(x, y, expr, param_list){ # object function
        x_pos <- length(param_list) + 1
        param_list[[x_pos]] <- x
        names(param_list)[x_pos] <- "x"
        return(sum((y - eval(expr, envir = param_list))^2))
}

grad <- function(x, y, expr, param_list){ # gradient based on sse
        x_pos <- length(param_list) + 1
        param_list[[x_pos]] <- x
        names(param_list)[x_pos] <- "x"
        param_names <- names(param_list)[-x_pos]
        deri_param <- deriv(f, param_names)
        return(-2 * t(attr(eval(deri_param, envir = param_list), "gradient")) %*% 
                       (y - eval(expr, envir = param_list)))
}


sgd <- function(x, y, expr, param_list, 
                alpha = 0.5, m = length(y)/5, eps1 = 1e-6,
                eps2 = 1e-6, maxit = 1e6){
        thetas <- unlist(as.list(param_list))
        params <- as.list(thetas)
        n <- 0
        obj_last <- obj(x, 
                        y, 
                        expr, 
                        as.list(thetas))
        g_history <- c(0, 0, 0, 0)
        theta_history <- thetas
        obj_history <- 0
        while(TRUE){
                sample_ind <- sample(1:1000, m)
                g <- numeric(length(thetas))
                g <- g + 1/m * grad(x[sample_ind], 
                                    y[sample_ind], 
                                    expr, 
                                    params)
                if (alpha * sqrt(sum(g^2)) <= eps1) break
                thetas <- thetas - alpha * as.numeric(g)
                params <- as.list(thetas)
                obj_cur <- obj(x, 
                               y, 
                               expr, 
                               as.list(thetas))
                if (abs(obj_cur - obj_last)/obj_last < eps2){
                        thetas <- thetas + alpha * as.numeric(g)
                        break
                }
                obj_last <- obj_cur
                n <- n + 1
                g_history <- rbind(g_history, t(g))
                obj_history <- c(obj_history, obj_last)
                theta_history <- rbind(theta_history, thetas)
                if (n >= maxit) break
        }
        return(list(theta = thetas, iter = n, obj_history = obj_history, 
                    g_history = g_history, theta_history = theta_history, 
                    obj_last = obj_last))
}
