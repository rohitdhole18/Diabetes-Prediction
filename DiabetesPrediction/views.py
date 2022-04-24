from django.shortcuts import render

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn import svm
from sklearn.model_selection import train_test_split



def home(request):
    return render(request,"home.html")

def predict(request):
    result1 = "None"
    return render(request,'predict.html',{"result":result1})

def result(request):
    dataset = pd.read_csv("D:\MINOR PROJECT\diabetes_dataset.csv")

    x = dataset.drop(columns='Outcome', axis=1)
    y = dataset['Outcome']
    scaler = StandardScaler()
    scaler.fit(x)
    standardized_data = scaler.transform(x)

    x = standardized_data
    y = dataset['Outcome']

    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, stratify=y, random_state=2)

    classifier = svm.SVC(kernel='linear')
    classifier.fit(x_train, y_train)

    val1 = float(request.GET['n1'])
    val2 = float(request.GET['n2'])
    val3 = float(request.GET['n3'])
    val4 = float(request.GET['n4'])
    val5 = float(request.GET['n5'])
    val6 = float(request.GET['n6'])
    val7 = float(request.GET['n7'])
    val8 = float(request.GET['n8'])

    input_data = ([[val1, val2, val3, val4, val5, val6, val7, val8]])

    # Changing the input data to numpy array
    input_data_as_numpy_array = np.asarray(input_data)

    # Reshape the array as we are predicting for onr instance
    input_data_reshaped = input_data_as_numpy_array.reshape(1, -1)

    # Standardize the input data
    std_data = scaler.transform(input_data_reshaped)
    # print(std_data)

    prediction = classifier.predict(std_data)
    # print(prediction)

    result1 = "None"
    if (prediction[0] == 0):
        result1 = "Negative"

    else:
        result1 = "Positive"

    return render(request,'predict.html',{"result":result1})