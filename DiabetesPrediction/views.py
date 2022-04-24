from django.shortcuts import render

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression


def home(request):
    return render(request,"home.html")

def predict(request):
    result1 = "None"
    return render(request,'predict.html',{"result":result1})

def result(request):
    dataset = pd.read_csv("D:\MINOR PROJECT\diabetes_dataset.csv")

    x = dataset.drop('Outcome',axis=1)
    y = dataset['Outcome']
    X_train, X_test,y_train,y_test = train_test_split(x,y, test_size = 0.2, stratify=y, random_state=2)

    model = LogisticRegression()
    model.fit(X_train,y_train)

    val1 = float(request.GET['n1'])
    val2 = float(request.GET['n2'])
    val3 = float(request.GET['n3'])
    val4 = float(request.GET['n4'])
    val5 = float(request.GET['n5'])
    val6 = float(request.GET['n6'])
    val7 = float(request.GET['n7'])
    val8 = float(request.GET['n8'])

    pred = model.predict([[val1,val2, val3, val4, val5, val6, val7, val8]])
    print("rohit .....................")
    print(pred)
    print(pred)
    result1 = "None"
    if pred==[1]:
        result1 = "Positive"
    else:
        result1 = "Negative"

    return render(request,'predict.html',{"result":result1})