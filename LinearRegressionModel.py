import pandas as pd
import numpy as np
import sklearn
from sklearn import linear_model
from sklearn.utils import shuffle
import pickle
#P.S: Linear regression is useful when feautres are correlated with each others to some degree
#This is a simple LR model with sklearn.
#"pickle" module is used to save our model and we can train eventually a lot of models

#We load our data first and we choose the attributes we want to use as features
data= pd.read_csv("C:/Users/pc/Documents/student.csv" , sep=";")

data= data[['G1','G2','G3','studytime','failures','absences']]


predict='G3'

#In each ligne we put a training example
X=np.array(data.drop(predict,1)) #feautres
y=np.array(data[predict])#labels

#Split Our data to training data(90%) and Testing data (10%)
x_train,x_test,y_train,y_test=sklearn.model_selection.train_test_split(X,y, test_size=0.1)

#Define the model we wil be using

model=linear_model.LinearRegression()

#Actually training our model now with the training set
model.fit(x_train,y_train)
#Test The accuracy of our model
acc=model.score(x_test,y_test)
print(acc)
#Coefficients of the model
print("slope", model.coef_)
print("intercept", model.intercept_)
#All what's left now is to actually use the model to predict
predictions=model.predict(x_test)

for i in range(len(predictions)):
    print("prediction:", round(predictions[i]), "real_value:",y_test[i])