#install.packages('caret')
library(caret)
#install.packages('plyr')
library(plyr)
#install.packages('doMC')
library(doMC)
library(kernlab)
library(e1071)
library(ROCR)
library(readr)
library(dplyr)
library(randomForest)
library(ggplot2)
library(Hmisc)
library(party)


#Random Forest
library(randomForest)


#data.ariff=read.csv("fraud_dataset.csv")
#data.ariff=read.csv("new_1_10000.csv")
data.ariff=read.csv("new_1_5000.csv")

#data.ariff = subset(data.ariff, isFraud == 1,)
#write.csv(data.ariff,"pattern_fraud.csv")

data.ariff <- na.omit(data.ariff)
data.ariff$isFraud = as.factor(data.ariff$isFraud)
data.ariff$type = as.factor(data.ariff$type)
data.ariff$nameOrig = as.factor(data.ariff$nameOrig)
data.ariff$nameDest = as.factor(data.ariff$nameDest)
data.ariff = subset(data.ariff, select = -c(X))
#View(data.ariff)

train_index <- createDataPartition(data.ariff$isFraud, times = 1, p = 0.8, list = F)
X_train <- data.ariff[train_index,]
X_test <- data.ariff[-train_index,]
y_train <- data.ariff$isFraud[train_index]
y_test <- data.ariff$isFraud[-train_index]


train.one = subset(X_train, isFraud==1)
train.zero = subset(X_train, isFraud==0)

train.one.new = train.one[sample(nrow(train.one), nrow(train.zero), replace=TRUE),]
X_train = rbind(train.zero, train.one.new)

# set random seed for model reproducibility
set.seed(1234)

# look at the data
glimpse(data.ariff)

#X_train_rf = subset(X_train, select = -c(oldbalanceOrg, nameDest, nameOrig))
#X_test_rf = subset(X_test, select = -c(oldbalanceOrg, nameDest, nameOrig))

X_train_rf = subset(X_train, select = -c(nameDest, nameOrig))
X_test_rf = subset(X_test, select = -c(nameDest, nameOrig))

library(xgboost)
library(doMC)
library(dplyr)

x_train_xgb = X_train_rf
x_test_xgb = X_test_rf

x_train_xgb$type = as.numeric(x_train_xgb$type)
x_test_xgb$type = as.numeric(x_test_xgb$type)
x_train_xgb$isFraud = as.numeric(x_train_xgb$isFraud)
x_test_xgb$isFraud = as.numeric(x_test_xgb$isFraud)

x_test_xgb$isFraud[x_test_xgb$isFraud == 1] = 0
x_test_xgb$isFraud[x_test_xgb$isFraud == 2] = 1

x_train_xgb$isFraud[x_train_xgb$isFraud == 1] = 0
x_train_xgb$isFraud[x_train_xgb$isFraud == 2] = 1

summary(x_test_xgb$isFraud)

train_xgb <- x_train_xgb[, !(colnames(x_train_xgb) %in% c("isFraud"))]
test_xgb <- x_test_xgb[, !(colnames(x_test_xgb) %in% c("isFraud"))]

summary(dtrain_X)


dtrain_X <- xgb.DMatrix(data = as.matrix(train_xgb), label = x_train_xgb$isFraud)
dtest_X <- xgb.DMatrix(data = as.matrix(test_xgb), label = x_test_xgb$isFraud)
xgb <- xgboost(data = dtrain_X, nrounds = 100, gamma = 0.1, max_depth = 10, objective = "binary:logistic", nthread = 7)

preds_xgb <- predict(xgb, dtest_X)

predict_test4 <- as.numeric(preds_xgb > 0.5)
#summary(as.factor(predict_test4))
#summary(as.factor(y_test))
confusionMatrix(as.factor(predict_test4), as.factor(x_test_xgb$isFraud))