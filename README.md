# Fraud Detection in Financial Transactions

The objective of the project is to predict whether a given financial transaction is fraud or not.

Fraudulent transactions are costing the financial industry a huge overhead and it is rising [[1]](#1). Given the nature of our course work and my desire to find employment in the finance industry, I believe working on a similar project will be immensely helpful.

## Data Source

I am planning on using the PaySim dataset from Kaggle.com [[2]](#2). The dataset is generated based on financial transactions collected over a month in Africa. It records individual transactions with different types. Given there are multiple sources, I selected this data source given that each transaction is labelled whether it is fraud or not. 

The data set contains 6M+ transactions and with 11 different columns. The following images from R shows the structure of the dataset.

![DataSet Structure](https://github.com/thilinimfdo/fraud_detection/blob/main/data/summary_dataset.png)

## Machine Learning

I plan to use supervised learning techniques to predict whether a given transaction is fraudulent or not.

- I first want to try out different analysis to select proper features. Such analysis would be statisical correlation, Gini Importance from Random Forest.
- With the help of filtered features, I would try different approaches such as Logistic Regression, SVM, Random Forest, XGBoost. 
- One of the challenges I need to overcome is the data set is highly skewed towards benign transactions and will make the model biased towards many false negatives.
- My two main objectives for ML selection would be to minimize false negatives (actual frauds getting tagged as benigns) and minized false positives (actual benign ones are getting tagged as frauds). The first case will decrease the effectivness of the ML while the later case will increase customer dis-statisfaction a lot when they get their genuine transactions blocked.
- I plan to use Python for ML modelling and R for correlation and initial feature selection.

## Database

![Entity Relationship Diagram](https://github.com/thilinimfdo/fraud_detection/blob/main/data/erd.jpeg)

The above Entity Relationship Diagram shows two main entities in the data set: Customer and the Transactions. Each transaction is coined with two customer data points: the cutomer sending the money (or originating the transaction) and the customer recieving the money. I plan to use PostGres SQL for this project.


## References
<a id="1">[1]</a> 
https://www.cnbc.com/2021/01/27/credit-card-fraud-is-on-the-rise-due-to-covid-pandemic.html

<a id="2">[2]</a> 
https://www.kaggle.com/datasets/ealaxi/paysim1


