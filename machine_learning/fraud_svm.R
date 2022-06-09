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

#accuracy 0.954, kappa 0.1032, 1=3
svm.model <- svm(isFraud ~ amount + oldbalanceOrg, data = X_train, kernel = "linear", cost = 0.5, gamma = 0.001)

#accuracy 0.929, kappa 0.1524, 1=7
svm.model <- svm(isFraud ~ amount + oldbalanceOrg + oldbalanceDest + type, data = X_train, kernel = "linear", cost = 0.5, gamma = 0.001)

#accuracy 0.799, kappa 0.0592, 1=8
svm.model <- svm(isFraud ~ amount + newbalanceOrig + type, data = X_train, kernel = "linear", cost = 10, gamma = 0.001)

#accuracy 0.993, kappa 0.2208, 1=1
svm.model <- svm(isFraud ~ amount + nameOrig + type, data = X_train, kernel = "linear", cost = 0.5, gamma = 0.001)

#accuracy 0.993, kappa 0.2208, 1=1
svm.model <- svm(isFraud ~ amount + nameOrig + type + nameDest + oldbalanceDest + newbalanceDest, data = X_train, kernel = "linear", cost = 0.5, gamma = 0.001)

#accuracy 0.954, kappa 0.1958, 1=6
svm.model <- svm(isFraud ~ newbalanceOrig + oldbalanceOrg + type, data = X_train, kernel = "linear", cost = 0.5, gamma = 0.1)

#accuracy 0.965, kappa 0.1352, 1=3
svm.model <- svm(isFraud ~ amount - newbalanceOrig, data = X_train, kernel = "linear", cost = 0.5, gamma = 0.001)

#accuracy 0.993, kappa 0.2208, 1=1
svm.model <- svm(isFraud ~ amount + nameOrig + type - newbalanceOrig, data = X_train, kernel = "linear", cost = 0.5, gamma = 0.001)

#accuracy 0.994, kappa 0.3981, 1=2
svm.model <- svm(isFraud ~ amount + nameOrig + type / (oldbalanceOrg + newbalanceOrig), data = X_train, kernel = "linear", cost = 0.5, gamma = 0.001)

#accuracy 0.993, kappa 0.2208, 1=1
svm.model <- svm(isFraud ~ nameOrig + type + oldbalanceOrg / (amount), data = X_train, kernel = "linear", cost = 0.5, gamma = 0.001)

predict_test <- predict(svm.model, subset(X_test, select = -c(isFraud)), type = "response")
confusionMatrix(y_test, predict_test)

### KSVM  ###

models_ksvmL <- ksvm(isFraud ~ ., data=X_train, prob.model=T,kernel = "rbfdot",C=5, sigma = 1)

#accuracy 0.936, kappa 0.1675, 1=7
models_ksvmL <- ksvm(isFraud ~ amount + oldbalanceOrg, data=X_train, prob.model=T,kernel = "rbfdot",C=5, sigma = 1)

#accuracy 0.98, kappa 0.3254, 1=5
models_ksvmL <- ksvm(isFraud ~ amount + oldbalanceOrg + oldbalanceDest + type, data=X_train, prob.model=T,kernel = "rbfdot",C=5, sigma = 1)

#accuracy 0.877, kappa 0.0611, 1=5
models_ksvmL <- ksvm(isFraud ~ amount + newbalanceOrig + type, data=X_train, prob.model=T,kernel = "rbfdot",C=5, sigma = 1)

#accuracy 0.993, kappa 0.2208, 1=1
models_ksvmL <- ksvm(isFraud ~ amount + nameOrig + type, data=X_train, prob.model=T,kernel = "rbfdot",C=5, sigma = 1)

#accuracy 0.816, kappa 0.0368, 1=5
models_ksvmL <- ksvm(isFraud ~ newbalanceOrig + oldbalanceOrg + type, data=X_train, prob.model=T,kernel = "rbfdot",C=5, sigma = 1)

#accuracy 0.863, kappa 0.0788, 1=7 , 0=1
models_ksvmL <- ksvm(isFraud ~ amount - newbalanceOrig, data=X_train, prob.model=T,kernel = "rbfdot",C=5, sigma = 1)

#accuracy 0.993, kappa 0.2208, 1=1 , 0=7
models_ksvmL <- ksvm(isFraud ~ amount + nameOrig + type - newbalanceOrig, data=X_train, prob.model=T,kernel = "rbfdot",C=5, sigma = 1)

#accuracy 0.919, kappa 0.1347, 1=7,  0=1
models_ksvmL <- ksvm(isFraud ~ amount + nameOrig + type + oldbalanceOrg + nameDest / newbalanceOrig, data=X_train, prob.model=T,kernel = "rbfdot",C=5, sigma = 1)

#accuracy 0.814, kappa 0.0362, 1=5,  0=3
models_ksvmL <- ksvm(isFraud ~ oldbalanceOrg - amount, data=X_train, prob.model=T,kernel = "rbfdot",C=5, sigma = 1)

#accuracy 0.636, kappa 0.0269, 1=8,  0=0
models_ksvmL <- ksvm(isFraud ~ newbalanceOrig - amount, data=X_train_ksvm, prob.model=T,kernel = "rbfdot",C=5, sigma = 1)

#accuracy 0.794, kappa 0.0403, 1=6,  0=2
models_ksvmL <- ksvm(isFraud ~ oldbalanceOrg - newbalanceOrig, data=X_train, prob.model=T,kernel = "rbfdot",C=5, sigma = 1)

#accuracy 0.957, kappa 0.1775, 1=5,  0=3
models_ksvmL <- ksvm(isFraud ~ (oldbalanceOrg - newbalanceOrig) + type + amount, data=X_train, prob.model=T,kernel = "rbfdot",C=5, sigma = 1)

predict_test <- predict(models_ksvmL, X_test_ksvm, type = "response")

predict_test[predict_test > 0.5] = 1
predict_test[predict_test <= 0.5] = 0
summary(as.factor(predict_test))
levels(y_test)

confusionMatrix(as.factor(predict_test), as.factor(X_test_ksvm$isFraud))
