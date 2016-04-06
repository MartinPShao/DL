setwd("~/Documents/git/DL")
load("./hw5data.RData")
h=function(a){
        1/(1+exp(-a))
}

x=as.matrix(cbind(1,rbind(minst0_train, minst9_train)))
y=x

set.seed(1)
#split the sample into 50% training and 50% test by 
#making a test dataset and removing those observations
#from the data.  Changing the proportion to large values to
#holdout most of the data usually has excellent results.
test=sample(1:nrow(x),size=round(nrow(x)*.50),replace=FALSE)
x.test=x[test,]
y.test=y[test,]
x=x[-test,]
y=y[-test,]

###Set tuning parameters###
lam=1
gam=0.001
J=50
m=1
rho=0.05
beta=0.1
###########################
p=ncol(x)
K=ncol(y)

c=1;maxiter=500

betas=matrix(runif((J+1)*K,-.01,.01),nrow=K,ncol=(J+1))
alphas=matrix(runif(J*p,-.01,.01),ncol=p,nrow=J)

MSEs=rep(0,maxiter)
MSE.converge=.4 ###cross entropy convergence criteria
start=Sys.time()
repeat{
        samp.index=sample(1:nrow(x), size=m, replace = FALSE)
        y.s=y[samp.index,]
        x.s=x[samp.index,]
        
        z=cbind(1,h(x.s%*%t(alphas)))
        y.p=z%*%t(betas)  #each column is for a k
        
        KL=-rho/(z[, 2:(J+1)]+1e-6) + (1 - rho)/(1 - z[, 2:(J+1)] + 1e-9)
        y.diff=y.s-y.p
        
        ghat.beta=-t(t(z)%*%y.diff)/m + 2*lam*betas
        ghat.alpha=-(t((y.diff%*%betas[,2:(J+1)]+beta*KL)*z[,2:(J+1)]*(1-z[,2:(J+1)]))%*%x.s)/m + 2*lam*alphas
        
        betas=betas-gam*ghat.beta
        alphas=alphas-gam*ghat.alpha
        
        z=cbind(1,h(x%*%t(alphas)))
        y.p=z%*%t(betas)  #each column is for a k
        
        MSEs[c]=sum((y-y.p)^2)/m
        
        
        cat("Iteration: ",c,", MSE=",MSEs[c], "\n",sep="")
        
        if(MSEs[c]<MSE.converge| c>=maxiter){
                cat("Iteration: ",c,", MSE=",MSEs[c], "\n",sep="")
                break()
        }
        c=c+1
}
end=Sys.time()
end-start


if (FALSE){
        if(c<maxiter)
        {
                plot(CE[200:(min(which(CE==0))-1)],type="l",ylab="Cross Entropy")
        }else{
                plot(CE[200:length(CE)],type="l",ylab="Cross Entropy")
        }
        abline(h=0)
        
        z.test=cbind(1,sapply(1:J, function(j) h(x.test%*%alphas[j,])))
        o.k=sapply(1:K,function(w) z.test%*%betas[w,])
        y.p.test=sapply(1:K, function(q) exp(o.k[,q])/apply(exp(o.k),1,sum))
        
        cat("Test Misclassification = ",
            sum(apply(y.test,1,which.max)!=apply(y.p.test,1,which.max)),"/",nrow(x.test)," = ",
            100*(1-sum(apply(y.test,1,which.max)==apply(y.p.test,1,which.max))/nrow(x.test)),
            "%","\n",sep="")
}


