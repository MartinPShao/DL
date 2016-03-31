# build the CNN function
cnn <- function(im, y, kern_init, alpha_init, beta_init, m, gamma, lambda, prec) {
  # define the function of hidden nodes
  zj <- function(alpha, x) {
    temp1 <- x%*%alpha
    RVAL <- 1/(1+exp(-temp1))
    return(RVAL)
  }
  # define function for convolution
  conv <- function(kern, xx) {
    # convert data.frame to matrix
    img <- matrix(as.numeric(xx), sqrt(length(xx)), sqrt(length(xx)), byrow = TRUE)
    img <- as.matrix(Matrix::bdiag(list(0, img, 0)))
    ni <- nj <- sqrt(length(xx))
    conv_im <- rep(list(matrix(0, ni, nj)), length(kern))
    # start convolution
    for (k in 1:length(kern)) {
      for (i in 1:ni) {
        for (j in 1:nj) {
          conv_im[[k]][i, j] <- sum(kern[[k]]*img[i:(i + 2), j:(j + 2)])
        }
      }
    }
    return(conv_im)
  }
  # max pooling
  pool <- function(im, dime, ReLU = TRUE) {
    pool_im <- matrix(0, dime, dime)
    len <- dim(im)[1]/dime
    for (i in 1:dime) {
      for (j in 1:dime) {
        pool_im[i, j] <- max(im[(1 + len*(i - 1)):(len*i), (1+len*(j - 1)):(len*j)])
        if (ReLU == TRUE) {
          pool_im[i, j] <- max(0, pool_im[i, j])
        }
      }
    }
    return(as.numeric(pool_im))
  }
  # get the largest 3 by 3 image after convolution
  get_max <- function(im, im_conv, m, dime) {
    max_im <- list()
    len <- dim(im_conv[[1]][[1]])[1]/dime
    for (k in 1:5) {
      # loop for each kernel
      im_kernel <- sapply(im_conv,"[[", k)
      max4 <- list()
      for (l in 1:m) {
        img <- matrix(as.numeric(im[l, ]), 28, 28, byrow = TRUE)
        check_im <- matrix(as.numeric(im_kernel[, l]), 28, 28, byrow = TRUE)
        # img <- as.matrix(Matrix::bdiag(list(0, img, 0)))
        # find the index of maximum
        temp <- matrix(0, 4, 9)
        for (j in 1:dime) {
          for (i in 1:dime) {
            mm <- check_im[(1 + len*(i - 1)):(len*i), (1+len*(j - 1)):(len*j)]
            ind <- matrix(which(mm == max(mm), arr.ind = TRUE)[1, ],1, 2) + 
              c(len*(i - 1), len*(j - 1))
            xrange <- (ind[1, 1]-1):(ind[1, 1]+1)
            yrange <- (ind[1, 2]-1):(ind[1, 2]+1)
            # check if the maximum is on the edge
            if (all(xrange >= 1 & xrange <= 28 & yrange >=1 & yrange <= 28)) {
              temp[(i+2*(j-1)), ] <- as.numeric(img[xrange, yrange])
            } else {
              if (any(xrange < 1)) {
                if (any(yrange < 1)) {
                  temp[(i+2*(j-1)), ] <- as.numeric(bdiag(list(0, img[xrange, yrange])))
                }
                if (any(yrange > 28)) {
                  temp[(i+2*(j-1)), ] <- as.numeric(rbind(0, cbind(img[xrange, yrange[-3]], 0)))
                }
              } 
              if (any(xrange > 28)) {
                if (any(yrange < 1)) {
                  temp[(i+2*(j-1)), ] <- as.numeric(rbind(cbind(0, img[xrange[-3], yrange]), 0))
                }
                if (any(yrange > 28)) {
                  temp[(i+2*(j-1)), ] <- as.numeric(bdiag(list(img[xrange[-3], yrange[-3]], 0)))
                }
              }
            } # end checking 
          } # end loop i
        } # end loop j
        max4[[l]] <- temp
      } # end loop l (minibatch)
      max_im[[k]] <- max4
    } # end loop k (kernel)
    return(max_im)
  }
  # create lists for saving parameters
  kern <- alpha <- beta <- qmse <- list()
  kern[[1]] <- kern_init
  alpha[[1]] <- alpha_init
  beta[[1]] <- beta_init
  stop_c <- TRUE
  n <- 1
  while(stop_c) {
    ind <- sample(1:dim(im)[1], m, replace = FALSE)
    mini_im <- im[ind, ]
    mini_y <- y[ind]
    mini_x <- matrix(0, m, 21)
    # convert the image to input data
    temp_im <- apply(mini_im, 1, conv, kern = kern[[n]])
    for (p in 1:m) {
      mini_x[p, ] <- c(1,sapply(temp_im[[p]], pool, dime = 2))
    }
    # generate yhat
    mini_z <-zj(alpha[[n]], mini_x)
    yhat <- 1/(1 + exp(-cbind(1, mini_z)%*%matrix(beta[[n]], 11, 1)))
    # calculate mse
    qmse[[n]] <- -sum(mini_y*log(yhat))
    # calculate gradient
    g_a <- g_b <- 0
    g_a <- -t(as.matrix(mini_x))%*%(as.matrix(mini_y - yhat)%*%beta[[n]][-1]*mini_z*
                                      (1-mini_z))/m + 2*lambda/m*alpha[[n]]
    g_b <- -t(mini_y - yhat)%*%cbind(1, mini_z)/m + 2*lambda*beta[[n]]/m
    # update parameters
    alpha[[n+1]] <- alpha[[n]] - gamma*g_a 
    beta[[n+1]] <- beta[[n]] - gamma*g_b
    g_coef <- -((mini_y-yhat)%*%beta[[n]][-1]*mini_z*(mini_z))%*%t(alpha[[n]][-1,])
    inputs <- get_max(mini_im,temp_im, m, 2)
    # Finally, update kernel parameters!!!!
    kern[[n+1]] <- kern[[n]]
    for (q in 1:5) {
      g_k <- rowMeans(sapply(1:m, function(i) {g_coef[i, (1 + 4*(q - 1)):(4*q)]%*%
          inputs[[q]][[i]]}))
      g_k <- matrix(g_k, 3, 3)
      kern[[n+1]][[q]] <- kern[[n]][[q]] - gamma*g_k
    } # end loop for updating kernel
    temp_im <- apply(mini_im, 1, conv, kern = kern[[n+1]])
    for (p in 1:m) {
      mini_x[p, ] <- c(1,sapply(temp_im[[p]], pool, dime = 2))
    }
    # generate yhat
    mini_z <-zj(alpha[[n+1]], mini_x)
    yhat <- 1/(1 + exp(-cbind(1, mini_z)%*%matrix(beta[[n+1]], 11, 1)))
    # calculate mse
    qmse[[n+1]] <- -sum(mini_y*log(yhat))
    stop_c <- abs(qmse[[n+1]]-qmse[[n]]) > prec*qmse[[n]]
    if (is.nan(stop_c)) {
      stop_c <- FALSE
    }
    n <- n + 1
  } # end loop for while
  RVAL <- list(alpha = alpha,
               beta = beta,
               kernel = kern,
               mse = qmse)
  return(RVAL)
}


# examples for cnn

library(Matrix)
library(data.table)
# construct the training set
trainx <- rbind(train0[1:100,],train9[1:100,])
trainy <- c(rep(0,100), rep(1,100))
kern_init <- rep(list(matrix(runif(9, -0.01, 0.01), 3, 3)), 5)
alpha_init <- matrix(runif(210, -0.01, 0.01), 21, 10)
beta_init <- runif(11, -0.01, 0.01)
m <- 100
gamma <- 0.5
lambda <- 0.001
prec <- 0.001
outcome1 <- cnn(im = trainx, y = trainy, kern_init = kern_init, 
                alpha_init = alpha_init,beta_init = beta_init,
                m = m, gamma = gamma, lambda = lambda,prec = prec
) 
plot.ts(outcome1$mse, xlab= "objective function", 
        main = "Trace plot of objective function")


#########################################################################

hybridABC <- function(y, L, burnin, eps) {
  x <- list()
  beta <- rep(0, L)
  sigma <- rep(0, L)
  # set up initial values
  x[[1]] <- matrix(sample(c(0,1), 64^2, replace = TRUE), 64, 64)
  beta[1] <- 0.5
  sigma[1] <- 2
  MH <- 0
  
  # sample x
  sample_x <- function(x, y, beta, sigma) {
    address <- expand.grid(x = 1:64, y = 1:64)
    # get neigbors index
    get.neighbors <- function(rw, mat) {
      # Relative addresses
      z <- rbind(c(0, -1, 1, 0),c(-1, 0, 0, 1))
      # Convert to absolute addresses 
      z2 <- t(z + unlist(rw))
      # Choose those with indices within mat 
      b.good <- rowSums(z2 > 0)==2  &  z2[,1] <= nrow(mat)  &  z2[,2] <=ncol(mat)
      mat[z2[b.good,]]
    }
    neib <- apply(address, 1, get.neighbors, mat = x)
    vec <- as.vector(x)
    count <- t(mapply(m = 1:4096, function(m) {c(sum(vec[m] == neib[[m]]),
                                                 sum(vec[m] != neib[[m]]))}))
    inv_x <- 1 - x
    h <- as.vector(-1/(2*sigma)*(y - x)^2) 
    invh <- as.vector(-1/(2*sigma)*(y - inv_x)^2)
    d <- beta*count[,1] + h
    inv_d <- beta*count[,2] + invh
    prob <- exp(apply(matrix(inv_d - d, ncol = 1), 1, function(x) min(x, 0)))
    U <- runif(4096)
    ind <- which(U < prob)
    vec[ind] <- 1 - vec[ind]
    newx <- matrix(vec, 64, 64) 
    return(newx)
  }
  # sample beta
  sample_beta <- function(beta, sig_beta, B, sigma, x, y, eps) {
    new_beta <- msm::rtnorm(1, mean = beta, sd = sqrt(sig_beta), lower = 0, upper = B )
    w <- sample_x(x, y, beta = new_beta, sigma = sigma)
    address <- expand.grid(x = 1:64, y = 1:64)
    # get neigbors index
    get.neighbors <- function(rw, mat) {
      # Relative addresses
      z <- rbind(c(0, -1, 1, 0),c(-1, 0, 0, 1))
      # Convert to absolute addresses 
      z2 <- t(z + unlist(rw))
      # Choose those with indices within mat 
      b.good <- rowSums(z2 > 0)==2  &  z2[,1] <= nrow(mat)  &  z2[,2] <=ncol(mat)
      mat[z2[b.good,]]
    }
    # calculate h(x)
    neibx <- apply(address, 1, get.neighbors, mat = x)
    vecx <- as.vector(x)
    countx <- mapply(m = 1:4096, function(m) {sum(vecx[m] == neibx[[m]])})
    hx <- sum(countx)
    # calculate h(w)
    neibw <- apply(address, 1, get.neighbors, mat = w)
    vecw <- as.vector(w)
    countw <- mapply(m = 1:4096, function(m) {sum(vecw[m] == neibw[[m]])})
    hw <- sum(countw)
    MHratio <- 0
    # MH                        
    if (abs(hx - hw) < eps* hx) {
      ratio <- dtnorm(beta, mean = new_beta, sd = sqrt(sig_beta), lower = 0, 
                      upper = B) / dtnorm(new_beta, mean = beta, 
                                          sd = sqrt(sig_beta), lower = 0, upper = B)
      u <- runif(1)
      if (u < ratio) {
        prop_beta <- new_beta
        MHratio <- 1
      } else {
        prop_beta <- beta
      }
    } else {
      prop_beta <- beta
    }
    RVAL <- list(prop_beta = prop_beta,
                 MHratio = MHratio)
    return(RVAL)
  }
  # start main MCMC
  for (t in 2:L) {
    x[[t]] <- sample_x(x = x[[t -1]], y = y, beta = beta[t -1], sigma = sigma[t - 1])
    sigma[t] <- rinvgamma(1, 3 + 4096/2, 1/(1 + 0.5*sum((y - x[[t]])^2)))
    temp <- sample_beta(beta = beta[t -1], sig_beta = 3, B = 5, sigma = sigma[t],
                        x = x[[t]], y = y, eps = eps)
    beta[t] <- temp$prop_beta
    if (t > (burnin + 1)) {
      MH <- MH + temp$MHratio  
    }
  }
  RVAL <- list(x = x[-(1:burnin)],
               beta = beta[-(1:burnin)],
               sigma = sigma[-(1:burnin)],
               ratio = MH/(L - burnin))
  return(RVAL)
}

# example for hybridABC
library(msm)
library(mvtnorm)
library(MCMCpack)
myst <- read.table("myst_im.dat")
post <- hybridABC(myst, 1000, 100, 0.001)
par(mfrow = c(2, 1))
plot.ts(post$beta, main = expression(paste("Trace plot for ", beta)), 
        ylab = expression(beta))
plot.ts(post$sigma, main = expression(paste("Trace plot for ", sigma^2)))
denoi <- Reduce("+", post$x)/900
