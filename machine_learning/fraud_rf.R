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