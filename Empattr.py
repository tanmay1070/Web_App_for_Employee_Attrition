import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pickle

emp_train=pd.read_csv("Train.csv") 
emp_test=pd.read_csv("Test.csv")

#Concatenating emp_train and emp_test into employees

employees = pd.concat([emp_train.drop(['Attrition_rate'], axis=1), emp_test])

print("employees.shape:",employees.shape)

#Checking for null values
employees.isnull().sum()

#So there are null values in 6 columns
#Here we have 3 options 
#1.Drop the only the null values
#2.Drop the entire rows 
#3.Numerical Imputation(Replace with mean,median,mode)

employees.info()

employees.columns

#Replace all null Values with median

employees['Age'].fillna(employees.Age.median(),inplace = True)
employees['Time_of_service'].fillna(employees.Time_of_service.median(),inplace=True)
employees['Pay_Scale'].fillna(employees.Pay_Scale.median(),inplace=True)
employees['Work_Life_balance'].fillna(employees.Work_Life_balance.median(),inplace=True)
employees['VAR2'].fillna(employees.VAR2.median(),inplace=True)
employees['VAR4'].fillna(employees.VAR4.median(),inplace=True)

employees.isnull().sum()

#Now all the null values have been replaced with their respective column's median
#Now seperating train and test set

employees_train = employees.iloc[0:7000]
employees_test = employees.iloc[7000:]

from sklearn.model_selection import train_test_split

X = employees_train.drop(['Employee_ID','Education_Level','Gender','Relationship_Status','Hometown','Unit','Decision_skill_possess','Travel_Rate','Post_Level','Compensation_and_Benefits','Work_Life_balance','VAR1','VAR2','VAR3','VAR4','VAR5'],axis=1).values
X_test = employees_test.drop(['Employee_ID','Education_Level','Gender','Relationship_Status','Hometown','Unit','Decision_skill_possess','Travel_Rate','Post_Level','Compensation_and_Benefits','Work_Life_balance','VAR1','VAR2','VAR3','VAR4','VAR5'],axis=1).values
y = emp_train['Attrition_rate'].values

#Applying Machine Learning Algo
#In this, we have have to predict Atrittion_rate(y) which is dependent on various other attributes(x) 
#So the best algo for the same is Linear Regression

from sklearn.linear_model import LinearRegression
clf = LinearRegression()
clf.fit(X, y)
Y_prediction = clf.predict(X_test)
print("Prediction:",Y_prediction)

#emp_test['Attrition_rate']=Y_prediction

#emp_test[['Employee_ID','Attrition_rate']].to_csv('submissionfinal.csv',index=False)

with open('Empattr.pkl','wb') as f:
	pickle.dump(clf,f)
