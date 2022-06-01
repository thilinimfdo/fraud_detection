
#data.ariff=read.csv("fraud_dataset.csv")
#data.ariff=read.csv("new_1_10000.csv")
data.ariff=read.csv("new_1_5000.csv")

#data.ariff = subset(data.ariff, isFraud == 1,)
#write.csv(data.ariff,"pattern_fraud.csv")

data.ariff <- na.omit(data.ariff)
data.ariff$isFraud = as.factor(data.ariff$isFraud)
data.ariff = subset(data.ariff, select = -c(X))
#View(data.ariff)

#install.packages('caret')
library(caret)
#install.packages('plyr')
library(plyr)
#install.packages('doMC')
library(doMC)
library(kernlab)
library(e1071)
library(ROCR)

#data.ariff.new = data.ariff
#data.ariff = data.ariff.new[1:5000,]
#write.csv(data.ariff, "new_1_5000.csv")
#rm(data.ariff.new)

for(l in names(data.ariff)) {
  if (is.numeric(data.ariff[[l]])) {
    data.ariff[[l]] = scale(data.ariff[[l]])
  }
}

train_index <- createDataPartition(data.ariff$isFraud, times = 1, p = 0.8, list = F)
X_train <- data.ariff[train_index,]
X_test <- data.ariff[-train_index,]
y_train <- data.ariff$isFraud[train_index]
y_test <- data.ariff$isFraud[-train_index]


# Parallel processing for faster training
registerDoMC(cores = 8)

# Use 10-fold cross-validation
ctrl <- trainControl(method = "cv",
                     number = 10,
                     verboseIter = T,
                     classProbs = T,
                     sampling = "smote",
                     summaryFunction = twoClassSummary,
                     savePredictions = T)

#log_mod <- glm(class ~ over_draft + foreign_worker + credit_usage, family = "binomial", data = X_train)
#summary(log_mod)
#predict_test <- as.numeric(predict(log_mod, X_test, type = "response") > 0.5)

#svm

train.one = subset(X_train, isFraud==1)
train.zero = subset(X_train, isFraud==0)

train.one.new = train.one[sample(nrow(train.one), nrow(train.zero), replace=TRUE),]
X_train = rbind(train.zero, train.one.new)

#accuracy 0.994, kappa 0.3981, 1=2
svm.model <- svm(isFraud ~ ., data = X_train, kernel = "linear", cost = 0.5, gamma = 0.001)
summary(svm.model)

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

## SETTING THE new dataset

X_train_ksvm = subset(X_train, select = -c(oldbalanceOrg, nameDest, nameOrig))
X_test_ksvm = subset(X_test, select = -c(oldbalanceOrg, nameDest, nameOrig))

library(kernlab)
library(e1071)

rbf = rbfdot(sigma = 0.01)
#accuracy 0.957, kappa 0.1775, 1=5,  0=3

models_ksvmL <- ksvm(isFraud ~ ., data=X_train_ksvm, prob.model=T,kernel = "rbfdot",C=5, sigma = 1 )

predict_test <- predict(models_ksvmL, X_test_ksvm, type = "response")

predict_test[predict_test > 0.5] = 1
predict_test[predict_test <= 0.5] = 0
summary(as.factor(predict_test))
levels(y_test)

confusionMatrix(as.factor(predict_test), as.factor(X_test_ksvm$isFraud))

#summary(as.factor(predict_test))
#summary(as.numeric(y_test))
#y_test = as.numeric(y_test) -1

# Use a threshold of 0.5 to transform predictions to binary
conf_mat <- confusionMatrix(as.factor(y_test), as.factor(predict_test))
summary(conf_mat)
print(conf_mat)

#XGBoost
#data.xgb = read.arff("credit_fruad.arff")
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

library(readr)
library(dplyr)
library(randomForest)
library(ggplot2)
library(Hmisc)
library(party)


#Random Forest
library(randomForest)
# set random seed for model reproducibility
set.seed(1234)

# look at the data
glimpse(data.ariff)

#X_train_rf = subset(X_train, select = -c(oldbalanceOrg, nameDest, nameOrig))
#X_test_rf = subset(X_test, select = -c(oldbalanceOrg, nameDest, nameOrig))

X_train_rf = subset(X_train, select = -c(nameDest, nameOrig))
X_test_rf = subset(X_test, select = -c(nameDest, nameOrig))

#write.csv(X_train_rf, "rf_train.csv")
#write.csv(X_test_rf, "rf_test.csv")

X_train_rf %>%
  select(isFraud) %>%
  group_by(isFraud) %>%
  summarise(count = n()) %>%
  glimpse

X_test_rf %>%
  select(isFraud) %>%
  group_by(isFraud) %>%
  summarise(count = n()) %>%
  glimpse

# build random forest model using every variable
#accuracy 0.995, kappa 0.7034, 1=6,  0=989
rfModel <- randomForest(isFraud ~ . , data = X_train_rf)
#summary(rfModel)

#rfModel <- randomForest(isFraud ~ . , data = X_train_rf, keep.forest=TRUE, importance=TRUE, ntree=100, replace=FALSE)

predicted <- predict(rfModel, X_test_rf)
#pred <- prediction(predicted, X_test_rf$isFraud)

library(caret)
#Accuracy 0.765 kappa 0.3436
confusionMatrix(predicted, X_test_rf$isFraud)

library(pROC)
roc_obj <- roc(as.numeric(X_test_rf$isFraud), as.numeric(predicted))
auc(roc_obj)

plot(sensitivity(predicted, X_test_rf$isFraud))
plot(specificity(predicted, X_test_rf$isFraud))
library(forecast)
plot(accuracy(predicted, X_test_rf$isFraud))
#roc value
plot(roc(as.numeric(predicted), as.numeric(X_test_rf$isFraud)))

#0.9989919
#sensitivity(predicted, X_test_rf$isFraud)
#0.75
#specificity(predicted, X_test_rf$isFraud)
# Area under the curve 0.9276
#roc(as.numeric(predicted), as.numeric(X_test_rf$isFraud))

library(MLmetrics)
F1_all <- F1_Score(X_test$isFraud, X_test$predicted)
F1_all

#print top 10 variables
options(repr.plot.width=5, repr.plot.height=4)
varImpPlot(rfModel,
           sort = T,
           n.var=10,
           main="Top 10 Most Important Variables")


#rfModelTrim1 <- randomForest(class ~  current_balance, 
 #                            data = X_train)
#X_test$predictedTrim1 <- predict(rfModelTrim1, X_test)
#F1_1 <- F1_Score(X_test$class, X_test$predictedTrim1)
#F1_1

X_train_rf = subset(X_train, select = -c(oldbalanceOrg, nameDest, nameOrig, step))
X_test_rf = subset(X_test, select = -c(oldbalanceOrg, nameDest, nameOrig, step))

install.packages(lme4)
library(lme4)
#### Logistic Regression ####
log_mod <- glm(isFraud ~ ., family = binomial(), data = X_train_rf)
#summary(log_mod)
predicted <- predict(log_mod, X_test_rf, type="response")

predicted[predicted > 0.5] = 1
predicted[predicted <= 0.5] = 0

library(caret)
#Accuracy 0.765 kappa 0.3436
confusionMatrix(as.factor(predicted), X_test_rf$isFraud)
summary(X_test_rf$isFraud)


