import csv, sqlite3

con = sqlite3.connect("data_fraud.db") # change to 'sqlite:///your_filename.db'
cur = con.cursor()
cur.execute("CREATE TABLE customer (id, balance, type);") # use your column names here
cur.execute("CREATE TABLE trans (id, time, origin, receiver, type, amount, fraud);") # use your column names here

with open('fraud_dataset.csv','r') as fin: # `with` statement available in 2.5+
    # csv.DictReader uses first line in file for column headings by default
    dr = csv.DictReader(fin) # comma is default delimiter
    i = 0
    for row in dr:
        add_origin = [row['nameOrig'], row['newbalanceOrig'], "c"]
        add_dest = [row['nameDest'], row['newbalanceDest'], "c"]
        add_transaction = [i, row['step'], row['nameOrig'], row['nameDest'], row['type'], row['amount'], row['isFraud']]
        cur.execute("INSERT INTO customer (id, balance, type) VALUES (?, ?, ?);", add_origin)
        cur.execute("INSERT INTO customer (id, balance, type) VALUES (?, ?, ?);", add_dest)
        cur.execute("INSERT INTO trans (id, time, origin, receiver, type, amount, fraud) VALUES (?, ?, ?, ?, ?, ?, ?);", add_transaction)
        i+=1

con.commit()
con.close()