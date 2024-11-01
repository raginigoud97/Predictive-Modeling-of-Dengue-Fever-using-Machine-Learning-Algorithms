from ast import alias
from concurrent.futures import process
from django.shortcuts import render

# Create your views here.
from django.shortcuts import render, HttpResponse
from django.contrib import messages

import Dengue_Disease_Predection

from .forms import UserRegistrationForm
from .models import UserRegistrationModel
from django.conf import settings
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.ticker as plticker
import datetime as dt
from sklearn import preprocessing, metrics
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.linear_model import LinearRegression
from sklearn import metrics
from sklearn.metrics import classification_report


# Create your views here.

def UserRegisterActions(request):
    if request.method == 'POST':
        form = UserRegistrationForm(request.POST)
        if form.is_valid():
            print('Data is Valid')
            form.save()
            messages.success(request, 'You have been successfully registered')
            form = UserRegistrationForm()
            return render(request, 'UserRegistrations.html', {'form': form})
        else:
            messages.success(request, 'Email or Mobile Already Existed')
            print("Invalid form")
    else:
        form = UserRegistrationForm()
    return render(request, 'UserRegistrations.html', {'form': form})

def UserLoginCheck(request):
    if request.method == "POST":
        loginid = request.POST.get('loginid')
        pswd = request.POST.get('pswd')
        print("Login ID = ", loginid, ' Password = ', pswd)
        try:
            check = UserRegistrationModel.objects.get(
                loginid=loginid, password=pswd)
            status = check.status
            print('Status is = ', status)
            if status == "activated":
                request.session['id'] = check.id
                request.session['loggeduser'] = check.name
                request.session['loginid'] = loginid
                request.session['email'] = check.email
                print("User id At", check.id, status)
                return render(request, 'users/UserHomePage.html', {})
            else:
                messages.success(request, 'Your Account Not at activated')
                return render(request, 'UserLogin.html')
        except Exception as e:
            print('Exception is ', str(e))
            pass
        messages.success(request, 'Invalid Login id and password')
    return render(request, 'UserLogin.html', {})


def UserHome(request):

    return render(request, 'users/UserHomePage.html', {})

def DatasetView(request):
    path = settings.MEDIA_ROOT + "//" + 'Dengue_Disease.csv'
    df = pd.read_csv(path, nrows=300)
    df = df.to_html
    return render(request, 'users/viewdataset.html', {'data': df})

# def ML(request):
#     import matplotlib.pyplot as plt
#     import numpy as np
#     import pandas as pd
#     #from sklearn.model_selection import train_test_split
#     from sklearn.model_selection import cross_val_score
#     from sklearn.model_selection import KFold
#     from sklearn.pipeline import Pipeline
#     from sklearn.preprocessing import MinMaxScaler
#     from tensorflow.python.keras.models import Sequential
#     from tensorflow.python.keras.layers import Dense    
#     from keras.wrappers.scikit_learn import KerasRegressor
#     #from sklearn.model_selection import train_test_split
#     from django.conf import settings
#     path = settings.MEDIA_ROOT + '//' + 'Dengue_Disease.csv'    
#     df = pd.read_csv(path)
#     df['Dengue'] = df['Dengue'].map({'Dengue': 1, 'noDengue': 0})
#     X = df.drop('Dengue',axis=1)
#     Y= df['Dengue']
#     print(Y)
#     from sklearn.model_selection import train_test_split
#     X_train, X_test, Y_train, Y_test = train_test_split(X,Y, test_size= 0.2, random_state=101)#shuffle=True
#     Y_train.value_counts()
#     Y_test.value_counts()
#     from sklearn.metrics import accuracy_score,precision_score,recall_score,f1_score

#     #algorithm-1
#     from sklearn.tree import DecisionTreeClassifier
#     model1 = DecisionTreeClassifier()
#     model1.fit(X_train, Y_train)
#     y_pred2 = model1.predict(X_test)
#     from sklearn.metrics import accuracy_score
#     accuracy1 = accuracy_score(Y_test, y_pred2) * 100
#     print('Accuracy1:', accuracy1)
#     from sklearn.metrics import precision_score
#     precision1 = precision_score(Y_test, y_pred2) * 100
#     print('precision1:',precision1)
#     from sklearn.metrics import recall_score
#     recall1 = recall_score(Y_test, y_pred2) * 100
#     print('recall1:',recall1)  
#     from sklearn.metrics import f1_score
#     f1score1 = f1_score(Y_test, y_pred) * 100
#     print('f1score1:',f1score1)

#     #algrothem-2
#     from sklearn.ensemble import GradientBoostingClassifier
#     model4= GradientBoostingClassifier()    #n_estimators=10,criterion='entropy'
#     model4.fit(X_train, Y_train)
#     y_pred4 = model4.predict(X_test)
#     from sklearn.metrics import accuracy_score
#     accuracy4 = accuracy_score(Y_test, y_pred4) * 100
#     print('Accuracy4:', accuracy4)
#     from sklearn.metrics import precision_score
#     precision4 = precision_score(Y_test, y_pred4) * 100
#     print('precision4:',precision4)
#     from sklearn.metrics import recall_score
#     recall4 = recall_score(Y_test, y_pred4) * 100
#     print('recall4:',recall4) 
#     from sklearn.metrics import f1_score
#     f1score4 = f1_score(Y_test, y_pred4) * 100
#     print('f1score4:',f1score4)
    
#     #algorithm-3
#     from xgboost import XGBClassifier
#     model5= XGBClassifier()    #n_estimators=10,criterion='entropy'
#     model5.fit(X_train, Y_train)
#     y_pred5 = model5.predict(X_test)
#     from sklearn.metrics import accuracy_score
#     accuracy5 = accuracy_score(Y_test, y_pred5) * 100
#     print('Accuracy5:', accuracy5)
#     from sklearn.metrics import precision_score
#     precision5 = precision_score(Y_test, y_pred5) * 100
#     print('precision5:',precision5)
#     from sklearn.metrics import recall_score
#     recall5 = recall_score(Y_test, y_pred5) * 100
#     print('recall5:',recall5) 
#     from sklearn.metrics import f1_score
#     f1score5 = f1_score(Y_test, y_pred5) * 100
#     print('f1score5:',f1score5)

def ml(request):
    import pandas as pd
    import matplotlib.pyplot as plt
    import seaborn as sns
    from sklearn.metrics import classification_report
    import os
    df = os.path.join(settings.MEDIA_ROOT, 'Dengue_Disease.csv' )

    df=pd.read_csv(df)
    print(df)

    df.shape
    df.isnull().sum()

    df.info()

    df.describe()

    df.Dengue.value_counts()

    df.Dengue.value_counts().plot(kind="bar", color=["brown", "yellow"])
    plt.show()

    X = df.drop('Dengue',axis=1)
    Y= df['Dengue']

    from sklearn.model_selection import train_test_split
    X_train, X_test, Y_train, Y_test = train_test_split(X,Y, test_size= 0.2, random_state=101)

    Y_train.value_counts()

    Y_test.value_counts()

    Y_test.value_counts()

    from sklearn.tree import DecisionTreeClassifier
    #from sklearn.neighbors import KNeighborsClassifier
    from sklearn.metrics import accuracy_score,confusion_matrix,precision_score
    # dt=DecisionTreeClassifier(criterion= 'gini', min_samples_split= 100,max_depth=20,max_features='sqrt', splitter= 'best',max_leaf_nodes=10,random_state=42)
    dt=DecisionTreeClassifier()
    dt.fit(X_train,Y_train)

    prediction=dt.predict(X_test)
    accuracy=accuracy_score(Y_test,prediction)*100
    from sklearn.metrics import precision_score
    precision= precision_score(Y_test, prediction) * 100
    print('precision4:',precision)
    from sklearn.metrics import recall_score
    recall= recall_score(Y_test, prediction) * 100
    print('recall4:',recall) 
    from sklearn.metrics import f1_score
    f1score= f1_score(Y_test, prediction) * 100
    print('f1score4:',f1score) 
    
    return render(request,'users/ml.html',{"accuracy": accuracy, "precision": precision, "recall":recall, 'f1score':f1score})


def prediction(request):
    import os
    if request.method == 'POST':
        import pandas as pd
        from sklearn.model_selection import train_test_split
        from django.conf import settings
        Age=request.POST.get("Age")  
        Sex=request.POST.get("sex")
        Fever=request.POST.get("Fever")
        Headache=request.POST.get("Headache")
        Arthralgia=request.POST.get("Arthralgia")
        Myalgia=request.POST.get("Myalgia")
        Jaundice=request.POST.get("Jaundice")
        Vomiting=request.POST.get("Vomiting")
        Ecchymosis=request.POST.get("Ecchymosis")
        meningitis=request.POST.get("meningitis")        
        Convulsions=request.POST.get("Convulsions")
        coma=request.POST.get("coma")
        IgM=request.POST.get("IgM")
        IgG=request.POST.get("IgG")
        path = settings.MEDIA_ROOT + '//' + 'Dengue_Disease.csv'
        df = os.path.join(settings.MEDIA_ROOT, 'Dengue_Disease.csv' )
        df=pd.read_csv(df)
        print(df)

        df.shape

        df.isnull().sum()

        df.info()

        df.describe()

        df.Dengue.value_counts()

        df.Dengue.value_counts().plot(kind="bar", color=["brown", "yellow"])
        plt.show()
        # df = df.drop(columns=[''])
        df = df.drop(columns=['Conjunctivitis or Pain behind eyes','Skin rash','Generalized weakness','Decrease of urine or anuria','Abdominal pain','watery diarrhea','Respiratory tract infection or respiratory insufficieny','Kidney failure'],axis=1)
       
        X = df.drop('Dengue',axis=1)
        Y= df['Dengue']

        from sklearn.model_selection import train_test_split
        X_train, X_test, Y_train, Y_test = train_test_split(X,Y, test_size= 0.2, random_state=101)

        Y_train.value_counts()

        Y_test.value_counts()

        Y_test.value_counts()

        from sklearn.tree import DecisionTreeClassifier
        from sklearn.ensemble import RandomForestClassifier
        #from sklearn.neighbors import KNeighborsClassifier
        from sklearn.metrics import accuracy_score,confusion_matrix,precision_score
        # dt=DecisionTreeClassifier(criterion= 'gini', min_samples_split= 100,max_depth=20,max_features='sqrt', splitter= 'best',max_leaf_nodes=10,random_state=42)
        dt=RandomForestClassifier()
        dt.fit(X_train,Y_train)
        
        test = [Age,Sex,Fever,Headache,Arthralgia,Myalgia,Jaundice,Vomiting,Ecchymosis,meningitis,Convulsions,coma,IgM,IgG]
        print(test)
        y_pred=dt.predict([test])

        print(y_pred)        
        if y_pred == 0:
            msg =  'Dengue Disease'
        else:
            msg =  ' no disease'
        return render(request,"users/prediction.html",{"msg":msg})
    else:
        return render(request,'users/prediction.html',{})
