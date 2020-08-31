from django.shortcuts import render
from django.http import HttpResponse
import pandas as pd
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler
import sklearn.linear_model
import sklearn.ensemble
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import VotingClassifier
from sklearn.tree import DecisionTreeClassifier
import joblib
import pickle

reLoadModel = joblib.load('./models/titanic_model.sav')
ens_Model = joblib.load('./models/ensemble_model.sav')
scaleee = pickle.load(open('./models/scaler.sav', 'rb'))

def index(request):

    context = ({'a':'Predicting the Titanic Survivors...'})
    return render(request, 'index.html', context)


def pipeline(td):
    #Label Encoder
    var_mod = ['Pclass','Sex','Embarked']
    lab_enc = LabelEncoder()
    for i in var_mod:
        td[i] = lab_enc.fit_transform(td[i])
    print(td)
    #One hot encoder
    td1 = pd.concat([td,pd.get_dummies(td['Sex'], prefix='gender')],axis=1)
    td1.drop(['Sex'],axis=1, inplace=True)
    td2 = pd.concat([td1,pd.get_dummies(td1['Pclass'], prefix='class')],axis=1)
    td2.drop(['Pclass'],axis=1, inplace=True)
    td3 = pd.concat([td2,pd.get_dummies(td2['Embarked'], prefix='Port')],axis=1)
    td3.drop(['Embarked'],axis=1, inplace=True)
    print(td3)
    #Scaling

    sc_X = StandardScaler()
    scaled_X_test = sc_X.fit_transform(td3)
    #predictions = ensemble.predict(scaled_X_test)
    return scaled_X_test





def pred(request):

    male =int(request.POST.get('male'))
    female =int(request.POST.get('female'))
    Age =int(request.POST.get('Age'))
    SibSp =int(request.POST.get('SibSp'))
    Parch =int(request.POST.get('Parch'))
    Fare =float(request.POST.get('Fare'))
    class1 =int(request.POST.get('class1'))
    class2 =int(request.POST.get('class2'))
    class3 =int(request.POST.get('class3'))
    port_c =int(request.POST.get('port_c'))
    port_q =int(request.POST.get('port_q'))
    port_s =int(request.POST.get('port_s'))

    temp={'Age':Age, 'SibSp':SibSp,'Parch':Parch,'Fare':Fare,'gender_female':female,'gender_male': male, 'class_1': class1,'class_2': class2,'class_3': class3, 'Port_C':port_c,'Port_Q':port_q,'Port_S':port_s}

    data_table = pd.DataFrame({'x':temp}).transpose()
    print(data_table)
    scaled_set = scaleee.transform(data_table)
    print(scaled_set)
    prediction = ens_Model.predict(scaled_set)[0]
    if prediction == 0:
        context = ({'b':'died!'})
    else:
        context = ({'b':'survived!'})
    return render(request, 'result.html', context)
