h=function(a){
  1/(1+exp(-a))
}

x=as.matrix(cbind(1,scale(read.table("H:/9100-Deep Learning/HW2/Wine_input.dat"))))
y=as.matrix(read.table("H:/9100-Deep Learning/HW2/Wine_output.dat"))

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
lam=0.0001
gam=.1
J=10
m=2
###########################
p=ncol(x)
K=ncol(y)

c=1;maxiter=20000

betas=matrix(runif((J+1)*K,-.01,.01),nrow=K,ncol=(J+1))
alphas=matrix(runif(J*p,-.01,.01),ncol=p,nrow=J)

CE=rep(0,maxiter)
CE.converge=.4 ###cross entropy convergence criteria
start=Sys.time()
repeat{
  samp.index=sample(1:nrow(x), size=m, replace = FALSE)
  y.s=y[samp.index,]
  x.s=x[samp.index,]
  
  z=cbind(1,h(x.s%*%t(alphas)))
  o.k=z%*%t(betas)  #each column is for a k
  y.p=exp(o.k)/apply(exp(o.k),1,sum)
  
  y.diff=y.s-y.p
  
  ghat.beta=-t(t(z)%*%y.diff)/m + 2*lam*betas
  ghat.alpha=-(t(y.diff%*%betas[,2:(J+1)]*z[,2:(J+1)]*(1-z[,2:(J+1)]))%*%x.s)/m + 2*lam*alphas
  
  betas=betas-gam*ghat.beta
  alphas=alphas-gam*ghat.alpha
  
  z=cbind(1,h(x%*%t(alphas)))
  o.k=z%*%t(betas)  #each column is for a k
  y.p=exp(o.k)/apply(exp(o.k),1,sum)
  
  CE[c]=-sum(apply(y*log(y.p),1,sum))
  
  if(c %% 100 ==0)
  {
    cat("Iteration: ",c,", CE=",CE[c],", Missclassification = ",
        sum(apply(y,1,which.max)!=apply(y.p,1,which.max)),"/",nrow(x)," = ",
        100*(1-sum(apply(y,1,which.max)==apply(y.p,1,which.max))/nrow(x)),"%",
        "\n",sep="")
  }
  
  if(CE[c]<CE.converge| c==maxiter){
    cat("Iteration: ",c,", CE=",CE[c],", Missclassification = ",
        sum(apply(y,1,which.max)!=apply(y.p,1,which.max)),"/",nrow(x)," = ",
        100*(1-sum(apply(y,1,which.max)==apply(y.p,1,which.max))/nrow(x)),"%",
        "\n",sep="")
    break()
  }
  c=c+1
}
end=Sys.time()
end-start



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

library(nnet)
cv.nnet <- nnet(x=x,y=y,size=10,decay=.0001,softmax=TRUE,MaxNWts = 10000)
cv.pred <- predict(cv.nnet, newdata=x.test,type="class")

sum(apply(y.test,1,which.max)==as.numeric(substr(cv.pred , start=2, stop=2)))
