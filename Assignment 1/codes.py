import pandas as pd
import numpy as np
import csv as csv
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB

data = pd.read_csv("train.csv")
test = pd.read_csv("test.csv")

data["Gender"]=np.where(data["Sex"]=="male",0,1)
test["Gender"]=np.where(test["Sex"]=="male",0,1)
data["Numbered_Embarked"]=np.where(data["Embarked"]=="S",0, np.where(data["Embarked"]=="C",1, np.where(data["Embarked"]=="Q",2,3)))
test["Numbered_Embarked"]=np.where(test["Embarked"]=="S",0, np.where(test["Embarked"]=="C",1, np.where(test["Embarked"]=="Q",2,3)))
median_age = data['Age'].dropna().median()
if len(data.Age[ data.Age.isnull() ]) > 0:
    data.loc[ (data.Age.isnull()), 'Age'] = median_age
median_fare = data['Fare'].dropna().median()
if len(data.Fare[ data.Fare.isnull() ]) > 0:
    data.loc[ (data.Fare.isnull()), 'Fare'] = median_fare
if len(test.Age[ test.Age.isnull() ]) > 0:
    test.loc[ (test.Age.isnull()), 'Age'] = median_age
if len(test.Fare[ test.Fare.isnull() ]) > 0:
    test.loc[ (test.Fare.isnull()), 'Fare'] = median_fare
data=data[["Survived", "Pclass", "Gender", "Age", "SibSp", "Parch", "Fare", "Numbered_Embarked"]].dropna(axis=0, how='any')
test=test[["PassengerId", "Pclass", "Gender", "Age", "SibSp", "Parch", "Fare", "Numbered_Embarked"]].dropna(axis=0, how='any')

model = GaussianNB()

used_features =["Pclass", "Gender", "Age", "SibSp", "Parch", "Fare", "Numbered_Embarked"]

model.fit(data[used_features].values,data["Survived"])

ids = test['PassengerId'].values

predictions = model.predict(test[used_features])

predictions_file = open("predictions.csv", "wt")
open_file_object = csv.writer(predictions_file)
open_file_object.writerow(["PassengerId","Survived"])
open_file_object.writerows(zip(ids, predictions))
predictions_file.close()


