# Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn

## AIM:
To write a program to implement the Decision Tree Classifier Model for Predicting Employee Churn.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm

1. Import the required libraries.
   
2. Upload and read the dataset.
   
3. Check for any null values using the isnull() function.
   
4. From sklearn.tree import DecisionTreeClassifier and use criterion as entropy.
   
5. Find the accuracy of the model and predict the required values by importing the required module from sklearn.

## Program:
```
Program to implement the Decision Tree Classifier Model for Predicting Employee Churn.
Developed by: S Kantha Sishanth
RegisterNumber: 212222100020
```
```py
import pandas as pd
df=pd.read_csv("Employee.csv")

df.head()
df.info()
df.isnull().sum()
df["left"].value_counts()

from sklearn.preprocessing import LabelEncoder
le=LabelEncoder()

df["salary"]=le.fit_transform(df["salary"])
df.head()

x=df[["satisfaction_level","last_evaluation","number_project","average_montly_hours","time_spend_company","Work_accident","promotion_last_5years","salary"]]
x.head()

y=df["left"]

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=100)

from sklearn.tree import DecisionTreeClassifier
dt=DecisionTreeClassifier(criterion="entropy")
dt.fit(x_train,y_train)
y_pred=dt.predict(x_test)

from sklearn import metrics
accuracy=metrics.accuracy_score(y_test,y_pred)
accuracy

dt.predict([[0.5,0.8,9,260,6,0,1,2]])

```

## Output:

### data.head()

![ml_1](https://github.com/Skanthasishanth/Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn/assets/118298456/2b255637-0e17-477e-bea6-c50b79a7c47a)

### data.info()

![ml_2](https://github.com/Skanthasishanth/Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn/assets/118298456/caf6d54a-0c7f-47a8-a4bc-cf0e38df9f85)

### isnull() and sum()

![ml_3](https://github.com/Skanthasishanth/Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn/assets/118298456/2518141a-c1a7-4da7-afc7-a980b0521c8f)

### data value counts()

![ml_4](https://github.com/Skanthasishanth/Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn/assets/118298456/2eb06960-84dd-4672-b461-ca6b15089962)

### data.head() for salary

![ml_5](https://github.com/Skanthasishanth/Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn/assets/118298456/07118456-06e3-4d6e-94cb-d659131b0b73)

### x.head()

![ml_6](https://github.com/Skanthasishanth/Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn/assets/118298456/450941b2-7d1e-4d73-96a6-eb8cca679dd7)

### accuracy value

![ml_7](https://github.com/Skanthasishanth/Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn/assets/118298456/80cae8c2-1b2b-4d28-9a37-1b4feeab35be)

### data prediction

![ml_8](https://github.com/Skanthasishanth/Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn/assets/118298456/4a303c1b-5989-4f51-8174-3ddc627b569d)


## Result:
Thus the program to implement the  Decision Tree Classifier Model for Predicting Employee Churn is written and verified using python programming.
