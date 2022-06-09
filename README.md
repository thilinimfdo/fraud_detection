# Fraud Detection in Financial Transactions

The objective of the project is to predict whether a given financial transaction is fraud or not.

Fraudulent transactions are costing the financial industry a huge overhead and it is rising [[1]](#1). Given the nature of our course work and my desire to find employment in the finance industry, I believe working on a similar project will be immensely helpful.

## Presentation

[Link to the Presentation](https://docs.google.com/presentation/d/1vcCLSUWVL2v4KA1WReFpHsrDAQz0dPHJ/edit?usp=sharing&ouid=118315510912750425598&rtpof=true&sd=true)

[Link to the Video](https://www.youtube.com/watch?v=886nIEKqzC8)

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

The entire data set of PaySim is 6M+ which was impossible to load and train the model. So I selected random 5000 data points and did the feature selection. Given the highly skewed dataset, my main selection criteria was the biasness of the model. I used Kappa value to measure the biasness. Anything below 0.5 resembles a biasness. Random Forest with 5 features produced the highest kappa value hence I selected Random Forest for the project. The following diagram shows the ROC graph for the selected RF model.

![ROC Graph](https://github.com/thilinimfdo/fraud_detection/blob/main/machine_learning/fraud_rf.png)

## Database

![Entity Relationship Diagram](https://github.com/thilinimfdo/fraud_detection/blob/main/data/erd.jpeg)

The above Entity Relationship Diagram shows two main entities in the data set: Customer and the Transactions. Each transaction is coined with two customer data points: the cutomer sending the money (or originating the transaction) and the customer recieving the money. I plan to use PostGres SQL for this project.

I have so far written the [script](https://github.com/thilinimfdo/fraud_detection/blob/main/data/csv_to_db.py) to transform the csv data to a sql lite database based on the above entity relationship diagram.

## Deployment

I used Python sklearn to implement Random Forest. A separate train module is embedded so that if there is no trained model, the code initiate a model.
I also used to one hot encoding to represent the categorical variable of payment type.
I used Python Flask to publish a rest API to call and detect whether a given transaction is fraudulent or not. The following image shows how to call the API and it returns 0 saying it is likely a benign transaction.

![API Call](https://github.com/thilinimfdo/fraud_detection/blob/main/api_service/rest_api.jpg)

We also deployed a web form so that people can use to enter the transaction data to see whether it is fraudulent or not.

![Web Form](https://github.com/thilinimfdo/fraud_detection/blob/main/api_service/form.jpg)

The web will return the following output saying whether its bening or fraud.

![Benign Response](https://github.com/thilinimfdo/fraud_detection/blob/main/api_service/benign_transaction.jpg)
![Fraud Response](https://github.com/thilinimfdo/fraud_detection/blob/main/api_service/fraudulent_transaction.jpg)

## Dashboard

[Dashboard on Balance vs Fraud](https://public.tableau.com/app/profile/thilini.fernando/viz/Fraud_count_circle/Dashboard1)

### Time and Fraud

[Tableau Link](https://public.tableau.com/app/profile/thilini.fernando/viz/FraudvsTime/Sheet1)

![Time vs Fraud](https://github.com/thilinimfdo/fraud_detection/blob/main/dashboard/Fraud_time.png)

### Fraud Ratio vs Benign Ratio

[Tableau Link](https://public.tableau.com/app/profile/thilini.fernando/viz/Fraud_count_circle/Amount_Fraud)

![Fraud vs Bening Ratio](https://github.com/thilinimfdo/fraud_detection/blob/main/dashboard/count_fraud.png)

### Closing balance of Receiver vs Fraud

[Tableau Link](https://public.tableau.com/app/profile/thilini.fernando/viz/Fraud_count_circle/Sheet4)

![Close Balance](https://github.com/thilinimfdo/fraud_detection/blob/main/dashboard/Destination_Balance_vs_Fraud.png)

### Fraud Amount vs Fraud

[Tableau link](https://public.tableau.com/app/profile/thilini.fernando/viz/Fraud_count_circle/Sheet3)

![Time vs Fraud](https://github.com/thilinimfdo/fraud_detection/blob/main/dashboard/Fraud_amount.png)

### Transaction Type vs Fraud

[Tableau Link](https://public.tableau.com/app/profile/thilini.fernando/viz/Fraud_Type/Fraud_vs_Type)

![Type vs Fraud](https://github.com/thilinimfdo/fraud_detection/blob/main/dashboard/Fraud_vs_Type.png)

### Opening Balance of Sender vs Fraud

[Tableau Link](https://public.tableau.com/app/profile/thilini.fernando/viz/Fraud_count_circle/Sheet5)

![Sender vs Fraud](https://github.com/thilinimfdo/fraud_detection/blob/main/dashboard/Old_Origin_vs_fraud.png)


## References
<a id="1">[1]</a> 
https://www.cnbc.com/2021/01/27/credit-card-fraud-is-on-the-rise-due-to-covid-pandemic.html

<a id="2">[2]</a> 
https://www.kaggle.com/datasets/ealaxi/paysim1


