import numpy as np
import pandas as pd
import sklearn
from sklearn.neighbors import KNeighborsClassifier
from sklearn import linear_model,preprocessing

data=pd.read_csv("C:/Users/pc/Desktop/ML in python/car.data")
print(data.head())
#We need to transform literal data to numerical data(low=0, medium=1, high=2) with preprocessing

#This is the preprocessor(Object)
#le=preprocessing.LabelEncoder()
#We can use np.unique() to know how many distinct values we have in the array
def convertListBuyingAndMaint(y):
    for j in range(len(y)):
        if(y[j]=="vhigh"):
            y[j]=3
        elif(y[j]=="high"):
            y[j]=2
        elif(y[j]=="med"):
            y[j]=1
        else:
            y[j]=0
    return y
def convertListDoorsAndPersons(y):
    for j in range(len(y)):
        if(y[j]=="2"):
            y[j]=2
        elif(y[j]=="3"):
            y[j]=3
        elif(y[j]=="4"):
            y[j]=4
        elif(y[j]=="5more" or y[j]=="more"):
            y[j]=5
    return y

def convertListSafety(y):
    for j in range(len(y)):
        if(y[j]=="high"):
            y[j]=2
        elif(y[j]=="med"):
            y[j]=1
        elif(y[j]=="low"):
            y[j]=0
    return y

def convertListLugBoot(y):
    for j in range(len(y)):
        if(y[j]=="big"):
            y[j]=2
        elif(y[j]=="med"):
            y[j]=1
        elif(y[j]=="small"):
            y[j]=0
    return y

def convertClass(y):
    for j in range(len(y)):
        if(y[j]=="unacc"):
            y[j]=0
        elif(y[j]=="acc"):
            y[j]=1
        elif(y[j]=="good"):
            y[j]=2
        elif(y[j]=="vgood"):
            y[j]=3
    return y

buying=list(data['buying'])
buying=convertListBuyingAndMaint(buying)

maint=list(data['maint'])
maint=convertListBuyingAndMaint(maint)


lug_boot=list(data['lug_boot'])
lug_boot=convertListLugBoot(lug_boot)

safety=list(data['safety'])
safety=convertListSafety(safety)

door=list(data['door'])
door=convertListDoorsAndPersons(door)

persons=list(data['persons'])
persons=convertListDoorsAndPersons(persons)

cls=list(data['class'])
cls=convertClass(cls)

"""print(safety)
persons[0]=2
print(type(door))
print("door:",np.unique(door))
print("persons:",np.unique(persons))
print(np.unique(cls))
print(len(maint))
print(len(buying))
print(len(safety))
print(len(lug_boot))
print(len(door))
print(len(persons))"""

#All the list have the same dimension(length) so we can use zip() to combine data in a list of features where each index refers to a training set [(buying,maint,door,persons,lug_boot,safety

X=list(zip(buying,maint,door,persons,lug_boot,safety)) #features
y=list(cls) #label

"""print(X[0], y[0])
print(X[1])
print(X[2])
print(X[3])"""

#Spliting Data into train Data and test data

x_train, x_test, y_train,y_test=sklearn.model_selection.train_test_split(X,y,test_size=0.1)

#Define the model we'll be using

model=KNeighborsClassifier(n_neighbors=5)

#Train Model with training sey

model.fit(x_train,y_train)

#The mean of the true answers(predictions) we got
acc=model.score(x_test,y_test)
print(acc)


#Finally predict using x_test

predictions=model.predict(x_test)
names=["unacc","acc","good","vgood"]
for i in range(len(predictions)):
    print("Prediction:",names[predictions[i]],"########",  "Reality",names[y_test[i]])





