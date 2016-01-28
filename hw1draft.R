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
q1.x <- x
q1.y <- y

inits <- data.frame(theta0 = c(4.1, 20, -20, 0, 4), 
                    theta1 = c(-1.4, 20, -20, 0, -1.5), 
                    theta2 = c(0.65, 20, -20, 0, 0.6), 
                    theta3 = c(0.35, 20, -20, 0, 0.4))

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
        deri_param <- deriv(expr, param_names)
        return(-2 * t(attr(eval(deri_param, envir = param_list), "gradient")) %*% 
                       (y - eval(expr, envir = param_list)))
}


sgd <- function(x, y, expr, param_list, 
                alpha = 0.5, m = length(y)/5, eps1 = 1e-6,
                eps2 = 1e-6, maxit = 1e4){
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
                if (alpha * sqrt(sum(g^2)) <= eps1) {print(alpha);break}
                thetas <- thetas - alpha * as.numeric(g)
                params <- as.list(thetas)
                obj_cur <- obj(x, 
                               y, 
                               expr, 
                               as.list(thetas))
                if (abs(obj_cur - obj_last)/obj_last < eps2){
                        thetas <- thetas + alpha * as.numeric(g)
                        print(alpha)
                        break
                }
                obj_last <- obj_cur
                n <- n + 1
                print(n)
                g_history <- rbind(g_history, t(g))
                obj_history <- c(obj_history, obj_last)
                theta_history <- rbind(theta_history, thetas)
                if (n >= maxit) {print(alpha);break}
        }
        return(list(theta = thetas, iter = n, obj_history = obj_history, 
                    g_history = g_history, theta_history = theta_history, 
                    obj_last = obj_last))
}

q1.out.inits <- list()
for (i in 1:5){
        q1.out.inits[[i]] <- sgd(q1.x, q1.y, exprs, as.list(inits[1, ]))
}
q1.tb1 <- cbind(t(sapply(q1.out.inits, "[[", 1)), 
                sapply(q1.out.inits, "[[", 6), 
                sapply(q1.out.inits, "[[", 2))
rownames(q1.tb1) <- c("(4.1, -1.4, 0.65, 0.35)", 
                      "(20, 20, 20, 20)", 
                      "(-20, -20, -20, -20)", 
                      "(0, 0, 0, 0)", 
                      "(4, -1.5, 0.6, 0.4)")
colnames(q1.tb1)[5:6] <- c("SSE", "n.iter")


init <- list(theta0 = 4.1, theta1 = -1.4, theta2 = 0.65, theta3 = 0.35)
alpha <- seq(0.1, 0.9, 0.1)
q1.out.alphas <- list()
for (i in 1:5){
        q1.out.alphas[[i]] <- sgd(q1.x, q1.y, exprs, init, alpha = alpha[i])
}
q1.out.alphas[[6]] <- sgd(q1.x, q1.y, exprs, init, alpha = alpha[6])
q1.out.alphas[[7]] <- sgd(q1.x, q1.y, exprs, init, alpha = alpha[7])
q1.out.alphas[[8]] <- sgd(q1.x, q1.y, exprs, init, alpha = alpha[8])
q1.out.alphas[[9]] <- sgd(q1.x, q1.y, exprs, init, alpha = alpha[9])
q1.tb2 <- cbind(t(sapply(q1.out.alphas, "[[", 1)), 
                sapply(q1.out.alphas, "[[", 6), 
                sapply(q1.out.alphas, "[[", 2))
rownames(q1.tb2) <- paste("alpha=",as.character(alpha), sep = "")
colnames(q1.tb2)[5:6] <- c("SSE", "n.iter")

m <- c(1, 10, 100, 200, 500, 1000)
q1.out.ms <- list()
q1.out.ms[[1]] <- sgd(q1.x, q1.y, exprs, init, alpha = 0.1, m = m[1])
q1.out.ms[[2]] <- sgd(q1.x, q1.y, exprs, init, alpha = 0.1, m = m[2])
q1.out.ms[[3]] <- sgd(q1.x, q1.y, exprs, init, alpha = 0.1, m = m[3])
q1.out.ms[[4]] <- sgd(q1.x, q1.y, exprs, init, alpha = 0.1, m = m[4])
q1.out.ms[[5]] <- sgd(q1.x, q1.y, exprs, init, alpha = 0.1, m = m[5])
q1.out.ms[[6]] <- sgd(q1.x, q1.y, exprs, init, alpha = 0.1, m = m[6])
q1.tb3 <- cbind(t(sapply(q1.out.ms, "[[", 1)), 
                sapply(q1.out.ms, "[[", 6), 
                sapply(q1.out.ms, "[[", 2))
rownames(q1.tb3) <- paste("m=",as.character(m), sep = "")
colnames(q1.tb3)[5:6] <- c("SSE", "n.iter")








sgdm <- function(x, y, expr, param_list, 
                 alpha = 0.5, m = length(y)/5, eps1 = 1e-6,
                 eps2 = 1e-6, maxit = 1e6, v = numeric(length(param_list)), 
                 phi = 0.5){
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
                v <- phi * v - alpha * as.numeric(g)
                if (sum(v^2) <= eps1) break
                thetas <- thetas + v
                params <- as.list(thetas)
                obj_cur <- obj(x, 
                               y, 
                               expr, 
                               as.list(thetas))
                if (abs(obj_cur - obj_last)/obj_last < eps2){
                        thetas <- thetas - v
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

adasgd <- function(x, y, expr, param_list, 
                   alpha = 0.5, m = length(y)/5, eps1 = 1e-6,
                   eps2 = 1e-6, maxit = 1e6){
        thetas <- unlist(as.list(param_list))
        params <- as.list(thetas)
        n <- 0
        r <- numeric(length(thetas))
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
                r <- r + as.numeric(g)^2
                delta <- (alpha/sqrt(r)) * as.numeric(g)
                if (sum(delta^2) <= eps1) break
                thetas <- thetas - delta
                params <- as.list(thetas)
                obj_cur <- obj(x, 
                               y, 
                               expr, 
                               as.list(thetas))
                if (abs(obj_cur - obj_last)/obj_last < eps2){
                        thetas <- thetas + delta
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
q1.out.sgdm <- list() 
q1.out.sgdm[[1]] <- adasgd(q1.x, q1.y, exprs, init, alpha = 0.1, m = 200, 
                           maxit = 1e4, phi = 0.1)
q1.out.sgdm[[2]] <- sgdm(q1.x, q1.y, exprs, init, alpha = 0.1, m = 200, 
                         maxit = 1e4, phi = 0.5)
q1.out.sgdm[[3]] <- sgdm(q1.x, q1.y, exprs, init, alpha = 0.1, m = 200, 
                         maxit = 1e4, phi = 0.9)


q1.out.adasgd <- sgdm(q1.x, q1.y, exprs, init, alpha = 0.1, m = 200, 
                      maxit = 1e4)
q1.tb4 <- rbind(cbind(t(q1.out.ms[[4]][[1]]), 
                      q1.out.ms[[4]][[6]], 
                      q1.out.ms[[4]][[2]]), 
                cbind(t(sapply(q1.out.sgdm, "[[", 1)), 
                      sapply(q1.out.sgdm, "[[", 6), 
                      sapply(q1.out.sgdm, "[[", 2)), 
                cbind(t(q1.out.adasgd[[1]]), 
                      q1.out.adasgd[[6]], 
                      q1.out.adasgd[[2]]))
colnames(q1.tb4)[5:6] <- c("SSE", "n.iter")
rownames(q1.tb4) <- c("SGD", "SGD with momentum(0.1)", 
                      "SGD with momentum(0.5)", "SGD with momentum(0.9)", 
                      "AdaSGD")


## 2
n <- 500
p <- 3
theta <- matrix(c(0.3, 0.6, -0.3, 0.2))
x <- list()
pi <- list()
y <- list()

logis <- function(x){
        return(1/(1+exp(-x)))
}

for (i in 1:5){
        x[[i]] <- cbind(rep(1, n), matrix(rnorm(n*p), nrow = n, ncol = p))
        pi[[i]] <- logis(x[[i]] %*% theta)
        y[[i]] <- round(pi[[i]])
}
percent_one <- sapply(y, sum)/(n*p)


q2.x <- x[[1]]
q2.y <- y[[1]]
cost <- function(x, y, theta){
        n <- length(y)
        f <- 1/(1+exp(-(x %*% theta)))
        return(-1/n * (t(y) %*% matrix(log(f) + 1e-9) + 
                               t(1-y) %*% matrix(log(1 - f + 1e-9))))
}

grad_class <- function(x, y, theta){
        n <- length(y)
        f <- 1/(1+exp(-x %*% theta))
        p <- length(theta)
        grad <- numeric(p)
        for (i in 1:p){
                grad[i] <- -t((y - f)/(f*(1-f) + 1e-9)) %*% 
                        (x[,i]*exp(-x %*% theta)/(1+exp(-x %*% theta))^2)/n
        }
        return(grad)
}

sgd_class <- function(x, y, inits, alpha = 0.5 , m = length(y)/5, 
                      maxit = 1e5, eps1 = 1e-6, eps2 = 1e-3){
        thetas <- inits
        n <- 0
        obj_last <- cost(x, y, thetas)
        g_history <- c(0, 0, 0, 0)
        theta_history <- thetas
        obj_history <- obj_last
        while(TRUE){
                sample_ind <- sample(1:500, m)
                g <- numeric(length(thetas))
                g <- g + grad_class(x[sample_ind, ], 
                                    y[sample_ind, ], 
                                    thetas)
                if (alpha * sqrt(sum(g^2)) <= eps1) break
                thetas <- thetas - alpha * as.numeric(g)
                obj_cur <- cost(x, y, thetas)
                if (abs(obj_cur - obj_last)/obj_last < eps2){
                        thetas <- thetas + alpha * as.numeric(g)
                        break
                }
                obj_last <- obj_cur
                n <- n + 1
                g_history <- rbind(g_history, t(g))
                obj_history <- c(obj_history, obj_last)
                theta_history <- cbind(theta_history, thetas)
                if (n >= maxit) break
        }
        return(list(theta = thetas, iter = n, obj_history = obj_history, 
                    g_history = g_history, theta_history = theta_history, 
                    obj_last = obj_last))
}

inits <- data.frame(theta0 = c(0.4, 20, -20, 0, 0.3), 
                    theta1 = c(0.45, 20, -20, 0, 0.6), 
                    theta2 = c(-0.35, 20, -20, 0, -0.3), 
                    theta3 = c(0.25, 20, -20, 0, 0.2))
q2.out.inits <- list()
for (i in 1:5){
        q2.out.inits[[i]] <- sgd_class(q2.x, q2.y, unlist(inits[i, ]))
}
q2.tb1 <- cbind(t(sapply(q2.out.inits, "[[", 1)), 
                sapply(q2.out.inits, "[[", 6), 
                sapply(q2.out.inits, "[[", 2))
rownames(q2.tb1) <- c("(0.4, 0.45, -0.35, 0.25)", 
                      "(20, 20, 20, 20)", 
                      "(-20, -20, -20, -20)", 
                      "(0, 0, 0, 0)", 
                      "(0.3, 0.6, -0.3, 0.2)")
colnames(q2.tb1)[5:6] <- c("cross-entroy", "n.iter")

init <- c(theta0 = 0.3, theta1 = 0.6, theta2 = -0.3, theta3 = 0.2)
alpha <- seq(0.1, 0.9, 0.1)
q2.out.alphas <- list()
q2.out.alphas[[1]] <- sgd_class(q2.x, q2.y, init, alpha = alpha[1])
q2.out.alphas[[2]] <- sgd_class(q2.x, q2.y, init, alpha = alpha[2])
q2.out.alphas[[3]] <- sgd_class(q2.x, q2.y, init, alpha = alpha[3])
q2.out.alphas[[4]] <- sgd_class(q2.x, q2.y, init, alpha = alpha[4])
q2.out.alphas[[5]] <- sgd_class(q2.x, q2.y, init, alpha = alpha[5])
q2.out.alphas[[6]] <- sgd_class(q2.x, q2.y, init, alpha = alpha[6])
q2.out.alphas[[7]] <- sgd_class(q2.x, q2.y, init, alpha = alpha[7])
q2.out.alphas[[8]] <- sgd_class(q2.x, q2.y, init, alpha = alpha[8])
q2.out.alphas[[9]] <- sgd_class(q2.x, q2.y, init, alpha = alpha[9])
q2.tb2 <- cbind(t(sapply(q2.out.alphas, "[[", 1)), 
                sapply(q2.out.alphas, "[[", 6), 
                sapply(q2.out.alphas, "[[", 2))
rownames(q2.tb2) <- paste("alpha=",as.character(alpha), sep = "")
colnames(q2.tb2)[5:6] <- c("cross-entroy", "n.iter")

m <- c(10, 100, 200, 500)
q2.out.ms <- list()
q2.out.ms[[1]] <- sgd_class(q2.x, q2.y, init, alpha = 0.9, m = m[1])
q2.out.ms[[2]] <- sgd_class(q2.x, q2.y, init, alpha = 0.9, m = m[2])
q2.out.ms[[3]] <- sgd_class(q2.x, q2.y, init, alpha = 0.9, m = m[3])
q2.out.ms[[4]] <- sgd_class(q2.x, q2.y, init, alpha = 0.9, m = m[4])
q2.tb3 <- cbind(t(sapply(q2.out.ms, "[[", 1)), 
                sapply(q2.out.ms, "[[", 6), 
                sapply(q2.out.ms, "[[", 2))
rownames(q2.tb3) <- paste("m=",as.character(m), sep = "")
colnames(q2.tb3)[5:6] <- c("cross-entroy", "n.iter")








rmsprop <- function(x, y, inits, alpha = 0.5 , m = length(y)/5, rho = 0.9, 
                    phi = 0.5, v = numeric(length(inits)), maxit = 1e5, 
                    eps1 = 1e-6, eps2 = 1e-3){
        thetas <- inits
        k <- 0
        r <- numeric(length(thetas))
        obj_last <- cost(x, y, thetas)
        g_history <- c(0, 0, 0, 0)
        theta_history <- thetas
        obj_history <- obj_last
        while(TRUE){
                sample_ind <- sample(1:500, m)
                thetas <- thetas + phi * v
                g <- grad_class(x[sample_ind, ], 
                                y[sample_ind, ], 
                                thetas)
                r <- rho * r + (1 - rho) * g^2
                v <- phi * v - alpha/sqrt(r) * g
                if (sum(v^2) <= eps1) break
                thetas <- thetas + v
                obj_cur <- cost(x, y, thetas)
                if (abs(obj_cur - obj_last)/obj_last < eps2){
                        thetas <- thetas - v
                        break
                }
                obj_last <- obj_cur
                k <- k + 1
                #                 print(k)
                g_history <- rbind(g_history, t(g))
                obj_history <- c(obj_history, obj_last)
                theta_history <- cbind(theta_history, thetas)
                if (n >= maxit) break
        }
        return(list(theta = thetas, iter = k, obj_history = obj_history, 
                    g_history = g_history, theta_history = theta_history, 
                    obj_last = obj_last, r, v))
}
q2.out.rmsprops <- list()
for (i in 1:10){
        q2.out.rmsprops[[i]] <- rmsprop(q2.x, q2.y, init, alpha = 0.9, m = 100, 
                                        rho = 0.9, phi = 0.5, maxit = 1e4)
}
q2.tb4 <- cbind(1:10, 
                t(sapply(q2.out.rmsprops, "[[", 1)), 
                sapply(q2.out.rmsprops, "[[", 6), 
                sapply(q2.out.rmsprops, "[[", 2))
colnames(q2.tb4)[c(1,6:7)] <- c("#", "cross-entroy", "n.iter")

save(list = ls(), file = "./hw1output.RData")

