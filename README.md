# Fraud Detection in Financial Transactions

The objective of the project is to predict whether a given financial transaction is fraud or not.

Fraudulent transactions are costing the financial industry a huge overhead and it is rising [[1]](#1). Given the nature of our course work and my desire to find employment in the finance industry, I believe working on a similar project will be immensely helpful.

## Data Source

I am planning on using the PaySim dataset from Kaggle.com [[2]](#2). The dataset is generated based on financial transactions collected over a month in Africa. It records individual transactions with different types. Given there are multiple sources, I selected this data source given that each transaction is labelled whether it is fraud or not. 

The data set contains 6M+ transactions and with 11 different columns. The following images from R shows the structure of the dataset.

![DataSet Structure](https://github.com/thilinimfdo/fraud_detection/blob/main/data/summary_dataset.png)

## Machine Learning

I explored 4 different machine learning techniques but 5 different methodologies:
	- Logisitc Regression
	- SVM – Linear Kernel
	- SVM – Gaussian Kernal
	- XGBoost
	- Random Forest

For each techniques, I tried different feature combinations. For Random Forest, I had to drop two categorical columns with more than 5 categories.

### Model Selection

![Model Selection](https://github.com/thilinimfdo/fraud_detection/blob/main/machine_learning/comparison.jpg)

![Gini Importance](https://github.com/thilinimfdo/fraud_detection/blob/main/machine_learning/gini_importance.jpg)

The entire data set of PaySim is 6M+ which was impossible to load and train the model. So I selected random 5000 data points and did the feature selection. Given the highly skewed dataset, my main selection criteria was the biasness of the model. I used Kappa value to measure the biasness. Anything below 0.5 resembles a biasness. Random Forest with 5 features produced the highest kappa value hence I selected Random Forest for the project.

## Database

![Entity Relationship Diagram](https://github.com/thilinimfdo/fraud_detection/blob/main/data/erd.jpeg)

The above Entity Relationship Diagram shows two main entities in the data set: Customer and the Transactions. Each transaction is coined with two customer data points: the cutomer sending the money (or originating the transaction) and the customer recieving the money. I plan to use PostGres SQL for this project.

## Deployment

I used Python sklearn to implement Random Forest. A separate train module is embedded so that if there is no trained model, the code initiate a model.
I also used to one hot encoding to represent the categorical variable of payment type.
I used Python Flask to publish a rest API to call and detect whether a given transaction is fraudulent or not

## Dashboard

I plan to use Tableau to visualize the distribution of the benign and fraudulent transaction.
I also plan to create a webpage that uses the rest API where users can enter the value and get a response whether it’s a fraud or not.



## References
<a id="1">[1]</a> 
https://www.cnbc.com/2021/01/27/credit-card-fraud-is-on-the-rise-due-to-covid-pandemic.html

<a id="2">[2]</a> 
https://www.kaggle.com/datasets/ealaxi/paysim1


