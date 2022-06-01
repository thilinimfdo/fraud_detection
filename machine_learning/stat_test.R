data.ariff=read.csv("fraud_dataset.csv")

df.fraud = subset(data.ariff, isFraud == 1,)
df.legitimate = subset(data.ariff, isFraud == 0,)

#Fraud - median of amount = 441423, mean of amount = 1467967 
summary(df.fraud)
#legitimate - median of amount = 74685, mean of amount = 178197 
summary(df.legitimate)

#clustering
library(tidyverse)
library(cluster)
library(factoextra)

df.fraud = subset(df.fraud, select = -c(step, type, nameOrig, nameDest, isFraud, isFlaggedFraud))
df.fraud.scale = scale(df.fraud)
distance = get_dist(df.fraud.scale)
fviz_dist(distance, gradient = list(low = "#00AFBB", mid = "white", high = "#FC4E07"))

#k2 = kmeans(df.fraud.scale, centers = 2, nstart = 25)
set.seed(123)
fviz_nbclust(df.fraud.scale, kmeans, method = "wss")
k2 = kmeans(df.fraud, centers = 4, nstart = 25)
str(k2)

#clustering full dataset

data.ariff = subset(data.ariff, select = -c(step, type, nameOrig, nameDest, isFlaggedFraud))
data.ariff$isFraud = as.numeric(data.ariff$isFraud)
kfull = kmeans(data.ariff, centers = 2, nstart = 25)

##########

data.ariff <- na.omit(data.ariff)
data.ariff$isFraud = as.factor(data.ariff$isFraud)
data.ariff$type = as.factor(data.ariff$type)
#data.ariff$nameDest = as.factor(data.ariff$nameDest)

#to get summary statistics
library(pastecs)
stat.desc(data.ariff)

wilcox.test(data.ariff$amount~data.ariff$isFraud)
wilcox.test(data.ariff$oldbalanceDest~data.ariff$isFraud)

library(coin)

wilcoxsign_test(data.ariff$amount~data.ariff$isFraud)
#Z = 2184.5, p-value < 2.2e-16
#alternative hypothesis: true mu is not equal to 0
#effect size 0.8660322
2184.5/sqrt(6362620)

wilcoxsign_test(data.ariff$oldbalanceOrg~data.ariff$isFraud)
#Z = 1714.9, p-value < 2.2e-16
#alternative hypothesis: true mu is not equal to 0
#effect size 0.6798621
1714.9/sqrt(6362620)

wilcoxsign_test(data.ariff$newbalanceDest~data.ariff$isFraud)
#Z = 1553.2, p-value < 2.2e-16
#alternative hypothesis: true mu is not equal to 0
#effect size 0.615757
1553.2/sqrt(6362620)

wilcoxsign_test(data.ariff$oldbalanceDest~data.ariff$isFraud)
#Z = 1408.7, p-value < 2.2e-16
#alternative hypothesis: true mu is not equal to 0
#effect size 0.5584709
1408.7/sqrt(6362620)

wilcoxsign_test(data.ariff$newbalanceOrig~data.ariff$isFraud)
#Z = 796.62, p-value < 2.2e-16
#alternative hypothesis: true mu is not equal to 0
#effect size 0.3158153
796.62/sqrt(6362620)

kruskal.test(data.ariff$amount~data.ariff$isFraud)
#Kruskal-Wallis chi-squared = 8273.4, df = 1, p-value < 2.2e-16

kruskal.test(data.ariff$oldbalanceOrg~data.ariff$isFraud)
#Kruskal-Wallis chi-squared = 9892.2, df = 1, p-value < 2.2e-16

kruskal.test(data.ariff$newbalanceDest~data.ariff$isFraud)
#Kruskal-Wallis chi-squared = 170.84, df = 1, p-value < 2.2e-16

kruskal.test(data.ariff$oldbalanceDest~data.ariff$isFraud)
#Kruskal-Wallis chi-squared = 1869.5, df = 1, p-value < 2.2e-16

kruskal.test(data.ariff$newbalanceOrig~data.ariff$isFraud)
#Kruskal-Wallis chi-squared = 4999.4, df = 1, p-value < 2.2e-16

library(lsr)
cramersV(data.ariff$type, data.ariff$isFraud)
#0.05891237

library(MASS)
tbl = table(data.ariff$type, data.ariff$isFraud)
chisq.test(tbl) 
#X-squared = 22083, df = 4, p-value < 2.2e-16


