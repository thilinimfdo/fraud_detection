import glob
import os
import sys
import threading
import time
import hashlib
import datetime
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score
from flask import Flask, request
from flask_restful import Resource, Api
from sqlalchemy import create_engine
from json import dumps
from flask_jsonpify import jsonify
#from sklearn.model_selection import train_test_split

app = Flask(__name__)
api = Api(app)

thread_local = threading.local()

def init_model():
    np.random.seed(2)
    train = pd.read_csv('rf_train.csv',  index_col=False )
    test = pd.read_csv('rf_test.csv',  index_col=False )

    labelencoder = LabelEncoder()
    train['type_cat'] = labelencoder.fit_transform(train['type'])
    test['type_cat'] = labelencoder.fit_transform(test['type'])

    cat = train[['type', 'type_cat']].drop_duplicates()
    print(cat)

    train.drop('type', axis=1, inplace=True)
    test.drop('type', axis=1, inplace=True)

    train_y = train.copy()
    test_y = test.copy()
    train.drop('isFraud', axis=1, inplace=True)
    test_y.drop('isFraud', axis=1, inplace=True)

    sc = StandardScaler()
    classifier = RandomForestClassifier(max_depth=10, random_state=0)
    classifier.fit(train, train_y['isFraud'])

    setattr(thread_local, "classifier", classifier)
    setattr(thread_local, "test_y", test_y)
    setattr(thread_local, "test", test)
    setattr(thread_local, "cat", cat)

    #print(test_y.loc[[1]])

def predict(amount, newbalanceOrig,  oldbalanceDest,  newbalanceDest,  typec):
    classifier = None
    try:
        classifier = getattr(thread_local, "classifier")
        test_y = getattr(thread_local, "test_y")
        test = getattr(thread_local, "test")
        cat = getattr(thread_local, "cat")
    except:
        print("attribute")

    if classifier is None:
        init_model()
        classifier = getattr(thread_local, "classifier")
        test_y = getattr(thread_local, "test_y")
        test = getattr(thread_local, "test")
        cat = getattr(thread_local, "cat")

    x = str(cat.loc[cat['type'] == typec]['type_cat']).split()[1]
    #print(x)

    test_y = pd.DataFrame(columns=test_y.columns)
    #type_cat = labelencoder.fit_transform(typec)
    new_row = {'amount':amount, 'newbalanceOrig':newbalanceOrig, 'oldbalanceDest':oldbalanceDest, 'newbalanceDest':newbalanceDest, 'type_cat': x}
    test_y = test_y.append(new_row, ignore_index=True)
    #print(test_y)

    pred_y = classifier.predict(test_y)
    #print(pred_y)
    return pred_y[[0]]
    #acc = accuracy_score(test['isFraud'], pred_y)
    #print(acc)


class Fraud_Name(Resource):
    def get(self, amount, new_balance_orig, old_balance_dest, new_balance_dest, typec):
        isFraud = predict(amount, new_balance_orig, old_balance_dest, new_balance_dest, typec)
        result = {'data': str(isFraud[0])}
        response = jsonify(result)
        response.headers.add('Access-Control-Allow-Origin', '*')
        return response

api.add_resource(Fraud_Name, '/isfraud/<amount>/<new_balance_orig>/<old_balance_dest>/<new_balance_dest>/<typec>') # Route_1


if __name__ == '__main__':
     app.run(port='5002', debug=True)
