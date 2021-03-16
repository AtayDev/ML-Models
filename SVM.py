import sklearn
from sklearn import datasets,linear_model
from sklearn import svm
#SVM is used for Classification Problems

#Loading Data
cancerData=datasets.load_breast_cancer()

#Features:
#print(cancerData.feature_names)
#Classes:
#print(cancerData.target_names)

X=cancerData.data #An array of Features Data[[5,3,5..],[1,..],..]
y=cancerData.target #An array of Classes Data[0,1,0,1,0,0,0,1,..]

#Spliting Data
x_train,x_test,y_train,y_test=sklearn.model_selection.train_test_split(X,y,test_size=0.1)

#Our SVM Model
#The kernel bring our dimensions up 1 To actually be able to classify the data correctly

model=svm.SVC(kernel="poly")

#Training Our Model
model.fit(x_train,y_train)
acc=model.score(x_test,y_test)
print(acc)
#Predict
predictions=model.predict(x_test)

for i in range (len(predictions)):
    print("Prediction:",predictions[i],"*****", "Reality:",y_test[i])







