library("AUC")
data <- read.csv("/home/steve/MEGA/independent_study/ind-stud-wn2019/compas-scores-two-years.csv")
data = subset(data, data$race=='African-American' | data$race=='Caucasian')
y = data$two_year_recid
z = as.numeric(factor(data$race))
x = model.matrix(two_year_recid ~ -1 + age * priors_count * juv_other_count 
                 * juv_fel_count * juv_misd_count * sex * race, data = data)
k = 10
# desired reduced-rank of postProcess(X)
# should be <= ncol(X)






################
# FUNCTIONS
################



# Calculates Aliverti et al's OG x_tilde matrix
aliverti_x_tilde <- function(X, Z, k)
{
  svdX = svd(X)
  V = svdX$u
  Ut = t(svdX$v)
  
  if (k < length(svdX$d)){
    for (i in k:length(svdX$d)){
      svdX$d[i] = 0
    }
  }
  SigmaTil = diag(svdX$d)
  
  temp = lm(V %*% SigmaTil ~ z)
  S = V %*% SigmaTil - cbind(1,z)%*%temp$coef
  return(S%*%Ut)
}



# Does what it says; takes probabilities from a predictive model,
# splits them based on the corresponding values of Z, and then
# plots the cdf of each set of values (on the same plot)
# ---REQUIRES THAT ENTRIES IN PARAMETERS CORRESPOND TO THE SAME OBSERVATIONS---
plot_empirical_class_cdf <- function(model.probs, z)
{
  prob0 = matrix(nrow = length(model.probs), ncol = 1)
  prob1 = matrix(nrow = length(model.probs), ncol = 1)
  
  for (i in 1:nrow(z)){
    if (z[i,] == 1) {
      prob1[i,] = model.probs[i]
    }
    else {
      prob0[i,] = model.probs[i]
    }
  }
  
  prob0 = sort(prob0, decreasing = F)
  prob1 = sort(prob1, decreasing = F)
  
  plot(prob0, pch = 20, col = 'blue')
  lines(prob0, pch = 20, col = 'blue')
  points(prob1, pch = 20, col = 'red')
  lines(prob1, pch = 20, col = 'red')
}



# Calculates and returns AUC, TPR, TNR, FPR, FNR, accuracy, prediction error
# ---REQUIRES THAT ENTRIES IN PARAMETERS CORRESPOND TO THE SAME OBSERVATIONS---
get_error_statistics <- function(y, y_hat, probs)
{
  labels = as.factor(as.matrix(y))
  rocurve = roc(probs, labels)
  aucurve = auc(rocurve)
  
  truePos = 0
  trueNeg = 0
  falsePos = 0
  falseNeg = 0
  rownames(confusionMatrix) <- c("positive (true,false)", 
                                 "negative (false,true)", 
                                 "(accuracy,errorRate)",
                                 "(null, AUC)")
  
  if (nrow(y) != length(y_hat)){
    print("Error in xunad")
  }
  
  for (i in 1:nrow(y)){
    if (y[i,] == 1 & y_hat[i] == 1){
      truePos = truePos + 1
    } 
    else if (y[i,] == 1 & y_hat[i] == 0){
      falseNeg = falseNeg + 1
    }
    else if (y[i,] == 0 & y_hat[i] == 1){
      falsePos = falsePos + 1
    }
    else{
      trueNeg = trueNeg + 1
    }
  }
  
  accuracy = (truePos + trueNeg) / (truePos + trueNeg + falsePos + falseNeg)
  errorRate = (falsePos + falseNeg) / (truePos + trueNeg + falsePos + falseNeg)
  
  confusionMatrix[1,1] = truePos / sum((y == 1) & (y_hat == 1))
  confusionMatrix[1,2] = falsePos / sum((y == 0) & (y_hat == 1))
  confusionMatrix[2,1] = falseNeg / sum((y == 1) & (y_hat == 0))
  confusionMatrix[2,2] = trueNeg / sum ((y == 0) & (y_hat == 0))
  confusionMatrix[3,1] = accuracy
  confusionMatrix[3,2] = errorRate
  confusionMatrix[4,2] = aucurve
  
  return(confusionMatrix)
}






################
# PROGRAM
################



# create rank-reduced x_tilde
xTilde = aliverti_x_tilde(x, z, k)

# create unadjusted rank-reduced x
SVD = svd(x)
if (k < length(SVD$d)){
  for (i in k:length(SVD$d)){
    SVD$d[i] = 0
  }
}
Sigma = diag(SVD$d)
xUnad = SVD$u %*% Sigma %*% t(SVD$v)



# split testing and training data
xtil.trainSize = floor(nrow(data) * 0.8)
xtil.testSize = nrow(data) - xtil.trainSize
xtil.trainSet = as.data.frame(head(xTilde, xtil.trainSize))
xtil.testSet = as.data.frame(tail(xTilde, xtil.testSize))

xunad.trainSize = floor(nrow(data) * 0.8)
xunad.testSize = nrow(data) - xunad.trainSize
xunad.trainSet = as.data.frame(head(xUnad, xunad.trainSize))
xunad.testSet = as.data.frame(tail(xUnad, xunad.testSize))

y.trainSize = floor(nrow(data) * 0.8)
y.testSize = nrow(data) - y.trainSize
y.trainSet = as.data.frame(head(y, y.trainSize))
y.testSet = as.data.frame(tail(y, y.testSize))

z.trainSize = floor(nrow(data) * 0.8)
z.testSize = nrow(data) - z.trainSize
z.trainSet = as.data.frame(head(z, z.trainSize))
z.testSet = as.data.frame(tail(z, z.testSize))



# fit the models, store probabilities and logit ouput
# (currently only logistic regression; will expand once I'm getting 
#  correct results for these models)
xtil.log.model = glm(as.matrix(y.trainSet) ~ as.matrix(xtil.trainSet), family = binomial(link='logit'))
xunad.log.model = glm(as.matrix(y.trainSet) ~ as.matrix(xunad.trainSet), family = binomial(link='logit'))

xtil.log.pred = predict(xtil.log.model, xtil.trainSet)
xtil.log.probs = predict(xtil.log.model, xtil.trainSet, type = 'response')
xunad.log.pred = predict(xunad.log.model, xunad.trainSet)
xunad.log.probs = predict(xunad.log.model, xunad.trainSet, type = 'response')



# using logit output, predict classes
length(xtil.log.pred)
for (i in 1:length(xtil.log.pred)){
  if (xtil.log.pred[i] > 0){
    xtil.log.pred[i] = 1
  } else { xtil.log.pred[i] = 0 }
  if (xunad.log.pred[i] > 0){
    xunad.log.pred[i] = 1
  } else { xunad.log.pred[i] = 0 }
}



# split model output based on Z to assess predictive bias
y.black.trainset = matrix(nrow=length(which(z.trainSet==1)), ncol=1)
xunad.black.logpred = matrix(nrow=length(which(z.trainSet==1)), ncol=1)
xunad.black.logprob = matrix(nrow=length(which(z.trainSet==1)), ncol=1)
xtil.black.logprob = matrix(nrow=length(which(z.trainSet==1)), ncol=1)
xtil.black.logpred = matrix(nrow=length(which(z.trainSet==1)), ncol=1)

y.white.trainset = matrix(nrow=length(which(z.trainSet==2)), ncol=1)
xunad.white.logpred = matrix(nrow=length(which(z.trainSet==2)), ncol=1)
xunad.white.logprob = matrix(nrow=length(which(z.trainSet==2)), ncol=1)
xtil.white.logprob = matrix(nrow=length(which(z.trainSet==2)), ncol=1)
xtil.white.logpred = matrix(nrow=length(which(z.trainSet==2)), ncol=1)

whitecount = 1
blackcount = 1

for (i in 1:nrow(z.trainSet)){
  print(i)
  if (z.trainSet[i,] == 1) {
    y.black.trainset[blackcount,] = y.trainSet[i,]
    xunad.black.logpred[blackcount,] = xunad.log.pred[i]
    xunad.black.logprob[blackcount,] = xunad.log.probs[i]
    xtil.black.logpred[blackcount,] = xtil.log.pred[i]
    xtil.black.logprob[blackcount,] = xtil.log.probs[i]
    blackcount = blackcount + 1
  }
  else {
    y.white.trainset[whitecount,] = y.trainSet[i,]
    xunad.white.logpred[whitecount,] = xunad.log.pred[i]
    xunad.white.logprob[whitecount,] = xunad.log.probs[i]
    xtil.white.logpred[whitecount,] = xtil.log.pred[i]
    xtil.white.logprob[whitecount,] = xtil.log.probs[i]
    whitecount = whitecount + 1
  }
}



# check ecdf and error statistics between models and races
plot_empirical_class_cdf(xunad.log.probs, z.trainSet)
plot_empirical_class_cdf(xtil.log.probs, z.trainSet)
xunad.black.errorstats = get_error_statistics(y.black.trainset, 
                                              xunad.black.logpred, 
                                              xunad.black.logprob)
xtil.black.errorstats = get_error_statistics(y.black.trainset, 
                                             xtil.black.logpred, 
                                             xtil.black.logprob)
xunad.white.errorstats = get_error_statistics(y.white.trainset, 
                                              xunad.white.logpred, 
                                              xunad.white.logprob)
xtil.white.errorstats = get_error_statistics(y.white.trainset, 
                                             xtil.white.logpred, 
                                             xtil.white.logprob)




################
# STORAGE
################





# Aliverti's OG code 
# Can't get it to work, says I'm passing in an x with infinite values, no matter 
# how hard I try. Plus, I don't understand this method of removing the effects 
# of Z, and my code should encode the equation in Lemma 2.1 exactly.

# OG = function(X, z, K, rescale = F) {
# 
#   # originally had parameter:
#   # K = max(2, round(NCOL(X)/10))
# 
#   if(!is.matrix(X)) stop("X must be a matrix")
#   if(rescale) X = scale(X)
#   SVD = svd(X)
#   temp = lm(SVD$u %*% diag(SVD$d[1:K]) ~ z)
#   S = SVD$u %*% diag(SVD$d[1:K]) - cbind(1,z)%*%temp$coef
#   return(list(S = S, U = t(SVD$v)))
# }
# xTilde = OG(x, z, k)