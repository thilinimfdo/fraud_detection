data.ariff=read.csv("new_1_5000.csv")

data.ariff <- na.omit(data.ariff)
data.ariff$isFraud = as.factor(data.ariff$isFraud)
data.ariff$type = as.factor(data.ariff$type)
data.ariff$nameOrig = as.factor(data.ariff$nameOrig)
data.ariff$nameDest = as.factor(data.ariff$nameDest)
data.ariff = subset(data.ariff, select = -c(X))
#View(data.ariff)

#tbl_type = data.frame(table(data.ariff$isFraud, data.ariff$type))
#tbl_type = data.frame(table(data.ariff$nameOrig))

#install.packages('caret')
library(caret)
#install.packages('plyr')
library(plyr)
#install.packages('doMC')
library(doMC)
library(kernlab)
library(e1071)
library(ROCR)

train_index <- createDataPartition(data.ariff$isFraud, times = 1, p = 0.8, list = F)
X_train <- data.ariff[train_index,]
X_test <- data.ariff[-train_index,]
y_train <- data.ariff$isFraud[train_index]
y_test <- data.ariff$isFraud[-train_index]

train.one = subset(X_train, isFraud==1)
train.zero = subset(X_train, isFraud==0)

train.one.new = train.one[sample(nrow(train.one), nrow(train.zero), replace=TRUE),]
X_train = rbind(train.zero, train.one.new)

log_mod <- glm(isFraud ~ amount + oldbalanceOrg, family = "binomial", data = X_train)

log_mod <- glm(isFraud ~ amount + oldbalanceOrg + oldbalanceDest + type, family = "binomial", data = X_train)

log_mod <- glm(isFraud ~ amount + newbalanceOrig + type, family = "binomial", data = X_train)

log_mod <- glm(isFraud ~ amount + nameOrig + type, family = "binomial", data = X_train)

log_mod <- glm(isFraud ~ amount + nameOrig + type + nameDest + oldbalanceDest + newbalanceDest, family = "binomial", data = X_train)

log_mod <- glm(isFraud ~ newbalanceOrig + oldbalanceOrg + type, family = "binomial", data = X_train)

log_mod <- glm(isFraud ~ amount - newbalanceOrig, family = "binomial", data = X_train)

log_mod <- glm(isFraud ~ amount + nameOrig + type - newbalanceOrig, family = "binomial", data = X_train)

log_mod <- glm(isFraud ~ amount + nameOrig + type / (oldbalanceOrg + newbalanceOrig), family = "binomial", data = X_train)

log_mod <- glm(isFraud ~ nameOrig + type + oldbalanceOrg / (amount), family = "binomial", data = X_train)

summary(log_mod)
predict_test <- as.factor(as.numeric(predict(log_mod, X_test, type = "response") > 0.5))
confusionMatrix(y_test, predict_test)
