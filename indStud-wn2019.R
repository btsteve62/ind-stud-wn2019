library(MatchIt)
library(AUC)
library(plyr)


# holy shit!!
# this automates formula generation!
#
# xnames <- paste(names(data[,colIndices]),sep = "")
# formula <- as.formula(paste("target ~ ", paste(xnames,collapse = "+")))


################
# TO-DO
################

# TROUBLESHOOT CDF PLOTS (AGAIN) - PROBLEM RETURNED AFTER MODEL PREDICTIONS
#   WERE MESSED WITH AND AFTER is_recid AND is_violent_recid WERE ADDED TO THE DATASET

# What could be the source of the differences between my OG output and Aliverti's
# output in their paper? We're also using a logistic model, with even less interaction
# terms... 
# --> Llack of dimensionality reduction in this example implementation. Check to confirm?





###################################################################################################
###################################################################################################
###################################################################################################


################
# FUNCTIONS
################




# new_x_tilde <-function(x, z, k) {
#   # binZ = ifelse(z > 0, 1, 0)
#   mod = lm(z ~ x)
#   beta = mod$coefficients[-c(1)]
#   print(sum(beta^2))
#   return (x - (z %*% t(beta) / (sum(beta^2))) )
#   
# }




aliverti_x_tilde <- function(X, z, k) {
  svdX = svd(X)
  V = svdX$u
  Ut = t(svdX$v)
  # if (k < length(svdX$d)){
  #   for (i in k:length(svdX$d)){
  #     svdX$d[i] = 0
  #   }
  # }
  SigmaTil = diag(svdX$d)
  temp = lm(V %*% SigmaTil %*% Ut ~ z)
  S = V %*% SigmaTil %*% Ut - cbind(1,z)%*%temp$coef
  return(S)
  # return(S%*%Ut)
}



# performs rank-reduction on X through singular value decomposition
# returned X_tilde matrix is of rank k (assuming k < ncol(X))
rank_reduce <- function(X, k)
{
  svdX = svd(X)
  V = svdX$u
  if (k < length(svdX$d)){
    for (i in k:length(svdX$d)){
      svdX$d[i] = 0
    }
  }
  return (V %*% diag(svdX$d))
}



# Takes model prediction values (logit output), and works backwards to return
# model probabilities (i.e. expit output)
get_prob_from_pred <- function(pred){
  prob = matrix(nrow = length(pred), ncol = 1)
  for (i in 1:length(pred)){
    prob[i] = (exp(pred[i]) / (1 + exp(pred[i])))
  }
  return(prob)
}


# Takes the expit output from logistic regression model output
# Returns the class that should be predicted based on expit
get_class_assignment <- function(pred){
  for (i in 1:length(pred)){
    if (pred[i] > 0.5){
      pred[i] = 1
    }
    else{
      pred[i] = 0
    }
  }
  return(pred)
}


# Does what it says; takes probabilities from a predictive model,
# splits them based on the corresponding class (values of Z), and then
# plots the cdf of each set of values on the same plot for comparison.
# 
# ---REQUIRES THAT ENTRIES IN PARAMETERS CORRESPOND TO THE SAME OBSERVATIONS---
plot_empirical_class_cdf <- function(model.probs, z, title)
{
  
  "
  An empirical cdf is different from a \"regular\" cdf in that it describes the
  distribution of an observed sample, as opposed to the distribution of a
  population. In both cases, though, it is the probability of observing a value
  less than or equal to some constant, within the sample or population. For
  example, if we have a sample of numbers X = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
  then the empirical cdf of X looks like this:

  P(X <= 0): 0
  P(X <= 1): 1/10
  P(X <= 2): 1/10 + 1/10 = 1/10 + P(X <= 1)
  P(X <= 3): 1/10 + 1/10 + 1/10 = 1/10 + P(X <= 2)
  ...
  For x in [1,10]:
  P(X <= x) = 1/10 + P(X <= (x-1))
  ...
  P(X > 10): 1
  "
  
  len = length(model.probs)
  prob0 = matrix(nrow = len, ncol = 1)
  prob1 = matrix(nrow = len, ncol = 1)
  index0 = 1
  index1 = 1
  # split by classification
  for (i in 1:length(z)){
    if (z[i] > 0) {
      prob1[index1] = model.probs[i]
      index1 = index1 + 1
    }
    else {
      prob0[index0] = model.probs[i]
      index0 = index0 + 1
    }
  }
  na.omit(prob0)
  na.omit(prob1)
  cdf0 = ecdf(prob0)
  cdf1 = ecdf(prob1)
  plot(cdf0, do.points=FALSE, col = 'grey', ylab = "ecdf", xlab = 'prob', main = title)
  plot(cdf1, do.points=FALSE, add=TRUE, col = 'black', ylab = 'ecdf', xlab = 'prob')
  legend("bottomright", c("black", "white"), col = c("black", "grey"), pch = 20)
}




# handles when Z is coded as {1,2} instead of {0,1}
# THIS VERSION IS FOR THE TREATMENT EFFECT ADJUSTMENT METHOD
plot_empirical_class_cdf_te <- function(model.probs, z, title)
{
  len = length(model.probs)
  prob0 = matrix(nrow = len, ncol = 1)
  prob1 = matrix(nrow = len, ncol = 1)
  index0 = 1
  index1 = 1
  # split by classification
  for (i in 1:length(z)){
    if (z[i] > 1) {
      prob1[index1] = model.probs[i]
      index1 = index1 + 1
    }
    else {
      prob0[index0] = model.probs[i]
      index0 = index0 + 1
    }
  }
  na.omit(prob0)
  na.omit(prob1)
  cdf0 = ecdf(prob0)
  cdf1 = ecdf(prob1)
  plot(cdf0, do.points=FALSE, col = 'grey', ylab = "ecdf", xlab = 'prob', main = title)
  plot(cdf1, do.points=FALSE, add=TRUE, col = 'black', ylab = 'ecdf', xlab = 'prob')
  legend("bottomright", c("black", "white"), col = c("black", "grey"), pch = 20)
}





# Calculates and returns AUC, TPR, TNR, FPR, FNR, accuracy, prediction error
# 
# ---REQUIRES THAT ENTRIES IN PARAMETERS CORRESPOND TO THE SAME OBSERVATIONS---
get_error_statistics <- function(y, y_hat, probs, z)
{
  y0 = matrix(nrow = length(y), ncol = 1)
  y1 = matrix(nrow = length(y), ncol = 1)
  yhat0 = matrix(nrow = length(y_hat), ncol = 1)
  yhat1 = matrix(nrow = length(y_hat), ncol = 1)
  prob0 = matrix(nrow = length(probs), ncol = 1)
  prob1 = matrix(nrow = length(probs), ncol = 1)
  index0 = 1
  index1 = 1
  
  # split by classification
  for (i in 1:length(z)){
    if (z[i] > 0) {
      y1[index1] = y[i]
      yhat1[index1] = y_hat[i]
      prob1[index1] = probs[i]
      index1 = index1 + 1
    }
    else {
      y0[index0] = y[i]
      yhat0[index0] = y_hat[i]
      prob0[index0] = probs[i]
      index0 = index0 + 1
    }
  }
  y0 = na.omit(y0)
  y1 = na.omit(y1)
  yhat0 = na.omit(yhat0)
  yhat1 = na.omit(yhat1)
  prob0 = na.omit(prob0)
  prob1 = na.omit(prob1)
  labels0 = as.factor(as.matrix(y0))
  labels1 = as.factor(as.matrix(y1))
  
  rocurve0 = roc(prob0, labels0)
  rocurve1 = roc(prob1, labels1)
  aucurve0 = auc(rocurve0)
  aucurve1 = auc(rocurve1)
  
  stats0 = get_matrix(y0, yhat0, aucurve0)
  stats1 = get_matrix(y1, yhat1, aucurve1)
  
  newList = list("white"=stats0, "black"=stats1)
  return(newList)
}


# Called from "get error stats" function; computes confusion matrix and returns
# a matrix that will display a table detailing a set of relevant statistics
get_matrix <-function(y, y_hat, aucurve)
{
  truePos = 0
  trueNeg = 0
  falsePos = 0
  falseNeg = 0
  sumy1 = 0
  sumy0 = 0
  sumyhat1 = 0
  sumyhat0 = 0
  confusionMatrix = matrix(nrow = 4, ncol = 2)
  rownames(confusionMatrix) <- c("positive (true,false)", 
                                 "negative (false,true)", 
                                 "(accuracy,classifError)",
                                 "(null, AUC)")
  if (length(y) != nrow(y_hat)){
    print("Error in xunad")
  }
  for (i in 1:length(y))
  {
    # calculate total
    if (y[i] == 1) { sumy1 = sumy1 + 1 }
    else { sumy0 = sumy0 + 1 }
    if (y_hat[i] == 1) { sumyhat1 = sumyhat1 + 1 }
    else { sumyhat0 = sumyhat0 + 1 }
    
    if (y[i] == 1 & y_hat[i,] == 1){
      truePos = truePos + 1
    } 
    else if (y[i] == 1 & y_hat[i,] == 0){
      falseNeg = falseNeg + 1
    }
    else if (y[i] == 0 & y_hat[i,] == 1){
      falsePos = falsePos + 1
    }
    else{
      trueNeg = trueNeg + 1
    }
  }
  accuracy = (truePos + trueNeg) / (truePos + trueNeg + falsePos + falseNeg)
  errorRate = (falsePos + falseNeg) / (truePos + trueNeg + falsePos + falseNeg)
  confusionMatrix[1,1] = truePos / (truePos + falseNeg) #TRUE POSITIVE RATE
  confusionMatrix[1,2] = 1 - confusionMatrix[1,1]       #FALSE POSITIVE RATE
  confusionMatrix[2,2] = trueNeg / (trueNeg + falsePos) #TRUE NEGATIVE RATE
  confusionMatrix[2,1] = 1 - confusionMatrix[2,2]       #FALSE NEGATIVE RATE
  confusionMatrix[3,1] = accuracy
  confusionMatrix[3,2] = errorRate
  confusionMatrix[4,2] = aucurve
  
  return(confusionMatrix)
}




# Calculates and returns AUC, TPR, TNR, FPR, FNR, accuracy, prediction error
# 
# ---REQUIRES THAT ENTRIES IN PARAMETERS CORRESPOND TO THE SAME OBSERVATIONS---
# THIS VERSION IS FOR THE TREATMENT EFFECT ADJUSTMENT METHOD
get_error_statistics_te <- function(y, y_hat, probs, z)
{
  y0 = matrix(nrow = length(y), ncol = 1)
  y1 = matrix(nrow = length(y), ncol = 1)
  yhat0 = matrix(nrow = length(y_hat), ncol = 1)
  yhat1 = matrix(nrow = length(y_hat), ncol = 1)
  prob0 = matrix(nrow = length(probs), ncol = 1)
  prob1 = matrix(nrow = length(probs), ncol = 1)
  index0 = 1
  index1 = 1
  
  # split by classification
  for (i in 1:length(z)){
    if (z[i] > 1) {
      y1[index1] = y[i]
      yhat1[index1] = y_hat[i]
      prob1[index1] = probs[i]
      index1 = index1 + 1
    }
    else {
      y0[index0] = y[i]
      yhat0[index0] = y_hat[i]
      prob0[index0] = probs[i]
      index0 = index0 + 1
    }
  }
  y0 = na.omit(y0)
  y1 = na.omit(y1)
  yhat0 = na.omit(yhat0)
  yhat1 = na.omit(yhat1)
  prob0 = na.omit(prob0)
  prob1 = na.omit(prob1)
  labels0 = as.factor(as.matrix(y0))
  labels1 = as.factor(as.matrix(y1))
  
  rocurve0 = roc(prob0, labels0)
  rocurve1 = roc(prob1, labels1)
  aucurve0 = auc(rocurve0)
  aucurve1 = auc(rocurve1)
  
  stats0 = get_matrix(y0, yhat0, aucurve0)
  stats1 = get_matrix(y1, yhat1, aucurve1)
  
  newList = list("white"=stats0, "black"=stats1)
  return(newList)
}


# Called from "get error stats" function; computes confusion matrix and returns
# a matrix that will display a table detailing a set of relevant statistics
#
# THIS VERSION IS FOR THE TREATMENT EFFECT ADJUSTMENT METHOD
get_matrix_te <-function(y, y_hat, aucurve)
{
  truePos = 0
  trueNeg = 0
  falsePos = 0
  falseNeg = 0
  sumy1 = 0
  sumy0 = 0
  sumyhat1 = 0
  sumyhat0 = 0
  confusionMatrix = matrix(nrow = 4, ncol = 2)
  rownames(confusionMatrix) <- c("positive (true,false)", 
                                 "negative (false,true)", 
                                 "(accuracy,classifError)",
                                 "(null, AUC)")
  if (length(y) != nrow(y_hat)){
    print("Error in xunad")
  }
  for (i in 1:length(y))
  {
    # calculate total
    if (y[i] == 1) { sumy1 = sumy1 + 1 }
    else { sumy0 = sumy0 + 1 }
    if (y_hat[i] == 1) { sumyhat1 = sumyhat1 + 1 }
    else { sumyhat0 = sumyhat0 + 1 }
    
    if (y[i] == 1 & y_hat[i,] == 1){
      truePos = truePos + 1
    } 
    else if (y[i] == 1 & y_hat[i,] == 0){
      falseNeg = falseNeg + 1
    }
    else if (y[i] == 0 & y_hat[i,] == 1){
      falsePos = falsePos + 1
    }
    else{
      trueNeg = trueNeg + 1
    }
  }
  accuracy = (truePos + trueNeg) / (truePos + trueNeg + falsePos + falseNeg)
  errorRate = (falsePos + falseNeg) / (truePos + trueNeg + falsePos + falseNeg)
  confusionMatrix[1,1] = truePos / (truePos + falseNeg) #TRUE POSITIVE RATE
  confusionMatrix[1,2] = 1 - confusionMatrix[1,1]       #FALSE POSITIVE RATE
  confusionMatrix[2,2] = trueNeg / (trueNeg + falsePos) #TRUE NEGATIVE RATE
  confusionMatrix[2,1] = 1 - confusionMatrix[2,2]       #FALSE NEGATIVE RATE
  confusionMatrix[3,1] = accuracy
  confusionMatrix[3,2] = errorRate
  confusionMatrix[4,2] = aucurve
  
  return(confusionMatrix)
}




# returns the difference between the elements of two
# 4x2 matrices (of error statistics)
get_disparity_matrix <- function(b.mat, w.mat) {
  
  dispMatrix = matrix(nrow = 4, ncol = 2)
  rownames(dispMatrix) <- c("positive (true,false)", 
                                 "negative (false,true)", 
                                 "(accuracy,classifError)",
                                 "(null, AUC)")
  for (i in 1:nrow(dispMatrix)){
    for (j in 1:ncol(dispMatrix)){
      dispMatrix[i,j] = b.mat[i,j] - w.mat[i,j]
    }
  }
  return(dispMatrix)
}






###################################################################################################
###################################################################################################
###################################################################################################
###################################################################################################
###################################################################################################
###################################################################################################
###################################################################################################
###################################################################################################
###################################################################################################
###################################################################################################
###################################################################################################
###################################################################################################


#################################################
### RECONSTRUCTION OF ALIVERTI ET AL.'S OG METHOD
#################################################


##################
# DATA PREPARATION
##################

# csv can be downloaded directly from https://github.com/propublica/compas-analysis
data <- read.csv("/home/steve/MEGA/1-independent_study/ind-stud-wn2019/compas-scores-two-years.csv")


# set up the data
dataSubset = subset(data, data$race=='African-American' | data$race=='Caucasian')
dataSubset$race = factor(dataSubset$race)
dataSubset$race = as.numeric(dataSubset$race)
dataSubset$sex = as.numeric(dataSubset$sex)
sumFem = sum(dataSubset$sex == 1)
sumMale = sum(dataSubset$sex == 2)
sumBlack = sum(dataSubset$race == 1)
sumWhite = sum(dataSubset$race == 2)
for (i in 1:length(dataSubset$race)){
  if (dataSubset$race[i] == 1){
    dataSubset$race[i] = 1/sumBlack
  }
  else if (dataSubset$race[i] == 2){
    dataSubset$race[i] = -1/sumWhite
  }
  if (dataSubset$sex[i] == 1){
    dataSubset$sex[i] = 1/sumFem
  }
  else if (dataSubset$sex[i] == 2){
    dataSubset$sex[i] = -1/sumMale
  }
}


# define main variables
# k = 50  #reduced-rank of postProcess(X); should be <= ncol(X)
z = dataSubset$race
y = dataSubset$two_year_recid
x = model.matrix(~ -1 + age +  priors_count + juv_other_count
                 + juv_fel_count + juv_misd_count + sex + is_recid + 
                   is_violent_recid + race, data = dataSubset)
summary(dataSubset)


##################
# MATRIX PROCESSING
##################


# create rank-reduced x_tilde matrices
xog = aliverti_x_tilde(x, z, k)


# create unadjusted rank-reduced x
SVD = svd(x)
# if (k < length(SVD$d)){
#   for (i in k:length(SVD$d)){
#     SVD$d[i] = 0 } }
Sigma = diag(SVD$d)
xun = SVD$u %*% Sigma %*% t(SVD$v)


##################
# TESTING
##################


# split testing and training data
x.trainSize = floor(nrow(x) * 0.8)
x.testSize = nrow(x) - x.trainSize
x.trainSet = head(x, x.trainSize)
x.testSet = tail(x, x.testSize)
xog.trainSize = floor(nrow(xog) * 0.8)
xog.testSize = nrow(xog) - xog.trainSize
xog.trainSet = head(xog, xog.trainSize)
xog.testSet = tail(xog, xog.testSize)
xun.trainSize = floor(nrow(xun) * 0.8)
xun.testSize = nrow(xun) - xun.trainSize
xun.trainSet = head(xun, xun.trainSize)
xun.testSet = tail(xun, xun.testSize)
y.trainSize = floor(length(y) * 0.8)
y.testSize = length(y) - y.trainSize
y.trainSet = head(y, y.trainSize)
y.testSet = tail(y, y.testSize)
z.trainSize = floor(length(z) * 0.8)
z.testSize = length(z) - z.trainSize
z.trainSet = head(z, z.trainSize)
z.testSet = tail(z, z.testSize)


# create logistic models, get probabilities and predictions
xog.model = glm.fit(xog.trainSet, y.trainSet, family = binomial(link = 'logit'))
xog.model$coefficients
xog.pred = xog.testSet %*% xog.model$coefficients
xog.prob = get_prob_from_pred(xog.pred)
xog.yhat = get_class_assignment(xog.prob)
xun.model = glm.fit(xun.trainSet, y.trainSet, family = binomial(link = 'logit'))
xun.model$coefficients
xun.pred = xun.testSet %*% xun.model$coefficients
xun.prob = get_prob_from_pred(xun.pred)
xun.yhat = get_class_assignment(xun.prob)

# interesting:
# xnew.pred = xnew.testSet %*% xnew.model$coefficients


##################
# GET RESULTS
##################


# plots and statistics
plot_empirical_class_cdf(xun.prob, z.testSet, "Unadjusted X Empirical Cdf (separated by Z)")
plot_empirical_class_cdf(xog.prob, z.testSet, "OG X_tilde Empirical Cdf (separated by Z)")
xog.errorStats = get_error_statistics(y.testSet, xog.yhat, xog.prob, z.testSet)
xun.errorStats = get_error_statistics(y.testSet, xun.yhat, xun.prob, z.testSet)
xun.diffs = get_disparity_matrix(xun.errorStats$black, xun.errorStats$white)
xog.diffs = get_disparity_matrix(xog.errorStats$black, xog.errorStats$white)






###################################################################################################
###################################################################################################
###################################################################################################
###################################################################################################
###################################################################################################
###################################################################################################
###################################################################################################
###################################################################################################
###################################################################################################
###################################################################################################
###################################################################################################
###################################################################################################


#################################################
### CONSTRUCTION OF NEW METHOD
#
# Note: If developed upon, the implementation of the algorithm should be 
#       packaged so that individual functions handle different parts and 
#       the parameter selection for matching (implemented later) can be 
#       automated as well.
#
#       Race starts out as a non-centered non-{0,1} binary variable to avoid
#       adjustments which cause the model to ignore Z (i.e. setting z-values = 0) 
#       A copy of the true race classification is kept, while the original race 
#       value is adjusted to counterbalance the "treatment effect" of having a 
#       non-baseline race designation. Interpretation of the predictors changes, 
#       however the between-group predictive output is far more balanced.
#################################################


##################
# DATA PREPARATION
##################

matchData <- subset(dataSubset, select =
                      c("two_year_recid", "race", "age", "priors_count",
                        "juv_other_count", "juv_fel_count", "juv_misd_count",
                        "sex", "is_recid", "is_violent_recid"))

# 1 is treatment (black), 0 is control (white)
matchData$binRace = ifelse(matchData$race > 0, 1, 0)
matchData$race = ifelse(matchData$race > 0, 2, 1)

model = glm(two_year_recid 
            ~ race + age + priors_count + juv_other_count +
              juv_fel_count + juv_misd_count + sex + is_recid + is_violent_recid,
            data = matchData, family = binomial)

before = matchData
after = before
after$trueZ = subset(data, data$race=="African-American" | data$race=="Caucasian")$race



##################
# DATA PROCESSING
#
# Note: In this example implementation, there are only two "levels" of Z (race).
#       Since Z is cast as a binary indicator (without factor level), we have
#       already chosen a control group and a treatment group; since there are
#       only these two groups to begin with, there is no need to set up a loop
#       structure to handle multiple "levels" of Z. So the first steps are already
#       complete, and we can move on to matching.
##################



m.out <- matchit(binRace ~ age +  priors_count + juv_other_count
                 + juv_fel_count + juv_misd_count + sex,
                 data = before)

    ###############################################################################
    # Note: In the future, this would be where matching "parameter" selection takes place.
    #       Ideally, multiple distance metrics and matching methods would be tried and
    #       cross validated, best combination for the specific dataset would be retained.
    #       For this application, full matching should be preferred, but may have to be
    #       balanced with generating enough matches so that the adjustment has a
    #       sufficient effect.
    ###############################################################################

      #################################################
      # ADJUST ON EACH OBSERVATION OF TREATMENT EFFECT
      #################################################
      # vvv WORKS
      #################################################

ctrlObs <- as.integer(row.names(m.out$match.matrix))
treatObs <- as.integer(m.out$match.matrix[,1])
sum(matchData$binRace[treatObs] == 1)
sum(matchData$binRace[ctrlObs] == 0)

for (j in 1:length(ctrlObs)) {

  treatOut = binomial()$linkinv(predict(model, before[treatObs[j],]))
  ctrlOut = binomial()$linkinv(predict(model, before[ctrlObs[j],]))
  te = (treatOut - ctrlOut)

  if(!is.na(treatOut) & !is.na(ctrlOut)){
    after$race[treatObs[j]] = before$race[treatObs[j]] * (1-te)
  }
}
# after = na.omit(after[which(is.na(after$race[treatObs])),])
      #################################################
      # ^^^ WORKS
      #################################################

      ####################################
      # ADJUST ON AVERAGE TREATMENT EFFECT
      ####################################
      # vvv DOESN'T WORK
      ####################################

# treatEffect = rep(NA, length(ctrlObs))
# 
# for (j in 1:length(ctrlObs)) {
#   
#   treatOut = binomial()$linkinv(predict(model, before[treatObs[j],]))
#   ctrlOut = binomial()$linkinv(predict(model, before[ctrlObs[j],]))
#   treatEffect[j] = (treatOut - ctrlOut)
# }
# 
# treatEffect = mean(na.omit(treatEffect))
# summary(treatEffect)
# after$race[treatObs] = before$race[treatObs] * (1-treatEffect)
# 
# for (j in 1:length(treatObs)) {
#   if(!is.na(treatOut[j]) & !is.na(ctrlOut[j])){
#     after$race[treatObs[j]] = before$race[treatObs[j]] * (1-treatEffect)
#   }
# }
# after = na.omit(after[which(is.na(after$race[treatObs])),])
      ####################################
      # ^^^ DOESN'T WORK
      ####################################


##################
# TESTING
##################


# split both into testing and training sets
b.trainsize = floor(nrow(before) * 0.8)
b.testsize = nrow(before) - b.trainsize
b.train = head(before, b.trainsize)
b.test = tail(before, b.trainsize)
a.trainsize = floor(nrow(after) * 0.8)
a.testsize = nrow(after) - a.trainsize
a.train = head(after, a.trainsize)
a.test = tail(after, a.trainsize)

# train logistic model for both, same as logistic model used in adjustment
b.model = glm(two_year_recid ~ race + age + priors_count + juv_other_count + 
                juv_fel_count + juv_misd_count + sex + is_recid + is_violent_recid, 
              data = b.train, family = binomial)
a.model = glm(two_year_recid ~ race + age + priors_count + juv_other_count + 
                juv_fel_count + juv_misd_count + sex + is_recid + is_violent_recid, 
              data = a.train, family = binomial)

# get and process model probs
b.pred = predict(b.model, b.test[,-c(1)], type = 'response')
b.prob = get_prob_from_pred(predict(b.model, b.test[,-c(1)]))
b.yhat = get_class_assignment(b.pred)
a.pred = predict(a.model, a.test[,-c(1)], type = 'response')
a.prob = get_prob_from_pred(predict(a.model, a.test[,-c(1)], type = 'response'))
a.yhat = get_class_assignment(a.pred)


##################
# GET RESULTS
##################


plot_empirical_class_cdf_te(b.pred, b.test$race, "ecdf of before probs")
plot_empirical_class_cdf_te(a.pred, a.test$race, "ecdf of after probs")
b.errorStats = get_error_statistics_te(b.test$two_year_recid, b.yhat, b.prob, b.test$race)
a.errorStats = get_error_statistics_te(a.test$two_year_recid, a.yhat, a.prob, a.test$race)
a.diffs = get_disparity_matrix(a.errorStats$black, a.errorStats$white)
b.diffs = get_disparity_matrix(b.errorStats$black, b.errorStats$white)



##################
# NOTES
##################
#
# This is a research report, but it is also an open proposal for future research. We ran out of time
# in this semester; the next step will be to implement a matching method using exact distance and
# matching on all non-protected attributes. If the results of adjustment using such a matching method
# are consistent with the results reported here, then the method could/should be developed further.
# Otherwise, a lack of improvement (or even worse results) will indicate that the assumptions and
# interpretations used thus far are violated or inaccurate (as the case may be), or that this approach
# to debiasing may not be a promising one for future work.
#
###################################################################################################
###################################################################################################
###################################################################################################
###################################################################################################
###################################################################################################
###################################################################################################
###################################################################################################
###################################################################################################
###################################################################################################
###################################################################################################
###################################################################################################
###################################################################################################


#################################################
### CHECK RESULTS FROM BOTH METHODS
#################################################


xun.diffs
xog.diffs
b.diffs
a.diffs

xun.errorStats$white
xun.errorStats$black
xog.errorStats$white
xog.errorStats$black

b.errorStats$white
b.errorStats$black
a.errorStats$white
a.errorStats$black

length(which(after$race != 1 & after$race != 2)) / length(after$race)
length(which(before$race == 2)) / length(before$race)
(0.28 * 0.60 * length(before$race)) / length(before$race)
# ^^^ interesting; ~28% of treatment observations were adjusted, which
#     accounts for only 16.8% of total observations (of which treated
#     observations themselves account for ~60%)

print("cool")
