-- Exported from QuickDBD: https://www.quickdatabasediagrams.com/
-- NOTE! If you have used non-SQL datatypes in your design, you will have to change these here.


CREATE TABLE "Customer" (
    "Customer_ID" str   NOT NULL,
    "Bank_Balance" float   NOT NULL,
    "Customer_Type" str   NOT NULL,
    CONSTRAINT "pk_Customer" PRIMARY KEY (
        "Customer_ID"
     )
);

CREATE TABLE "Transaction" (
    "Transaction_ID" int   NOT NULL,
    "Transaction_Time" time   NOT NULL,
    "Origin_ID" str   NOT NULL,
    "Reciever_ID" str   NOT NULL,
    "Transaction_Type" str   NOT NULL,
    "Transaction_Amount" float   NOT NULL,
    "Is_Fraud" int   NOT NULL,
    CONSTRAINT "pk_Transaction" PRIMARY KEY (
        "Transaction_ID"
     )
);

ALTER TABLE "Transaction" ADD CONSTRAINT "fk_Transaction_Origin_ID" FOREIGN KEY("Origin_ID")
REFERENCES "Customer" ("Customer_ID");

ALTER TABLE "Transaction" ADD CONSTRAINT "fk_Transaction_Reciever_ID" FOREIGN KEY("Reciever_ID")
REFERENCES "Customer" ("Customer_ID");

