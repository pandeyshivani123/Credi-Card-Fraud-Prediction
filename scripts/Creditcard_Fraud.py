#(1)LOADING DATA
#Import libraries
import pandas as pd
import numpy as np

#Load data files
credit_card_data = pd.read_csv("creditcard.csv")

#Sample of data
credit_card_data.head(10)

#Data info

print(credit_card_data.info())
#List of column names
list(credit_card_data)

#Types of data columns
credit_card_data.dtypes

#Summary statistics
credit_card_data.describe()

#Check for duplicate data
credit_card_data.duplicated().any

#(2)DATA CLEANING AND PREPROCESSING
#Find missing values
credit_card_data.isnull().sum()

#Impute missing values with mean (numerical variables)
credit_card_data.fillna(credit_card_data.mean(),inplace=True) 
credit_card_data.isnull().sum() 

credit_card_data.to_csv("cleaned_credit_card_fraud.csv", index=False)


from sklearn.preprocessing import StandardScaler
sc= StandardScaler()
credit_card_data['Amount']= sc.fit_transform(pd.DataFrame(credit_card_data['Amount']))

#distribution of legit and fraudulent transactions
credit_card_data['Class'].value_counts()

legit= credit_card_data[credit_card_data.Class==0]
fraud= credit_card_data[credit_card_data.Class==1]
print(legit.shape)
print(fraud.shape)
legit.Amount.describe()
fraud.Amount.describe()

# Compare the values for both transactions
credit_card_data.groupby('Class').mean()



#Build a sample dataset containing similar distribution of normal & fraud transactions 
#No. of fraud transaction-> 230

legit_sample=legit.sample(n=230)
#Concatenating two dataframes
new_dataset=pd.concat([legit_sample, fraud], axis=0)
new_dataset.head(5)
new_dataset.tail(5)
new_dataset['Class'].value_counts()

new_dataset.groupby('Class').mean()

#splitting the data into features and targets
X=new_dataset.drop('Class', axis=1)
Y=new_dataset['Class']

#Split the data into train and test data
from sklearn.model_selection import train_test_split
x_train,x_cv,y_train,y_cv = train_test_split(X,Y,test_size=0.2,stratify=Y, random_state=2)

print(X.shape, x_train.shape, x_cv.shape)

#Model training

#(a)LOGISTIC REGRESSION ALGORITHM
#Fit model
from sklearn.linear_model import LogisticRegression
model=LogisticRegression()
model.fit(x_train,y_train)

#Predict values for cv data
pred_cv=model.predict(x_cv)

#Evaluate accuracy of model
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
accuracy_score(y_cv,pred_cv) #92.39%
matrix=confusion_matrix(y_cv,pred_cv)
print(matrix)

#(b)DECISION TREE ALGORITHM
#Fit model
from sklearn import tree
dt=tree.DecisionTreeClassifier(criterion='gini')
dt.fit(x_train,y_train)

#Predict values for cv data
pred_cv1=dt.predict(x_cv)

#Evaluate accuracy of model
accuracy_score(y_cv,pred_cv1) #86.96%
matrix1=confusion_matrix(y_cv,pred_cv1)
print(matrix1)

#(c)RANDOM FOREST ALGORITHM
#Fit model
from sklearn.ensemble import RandomForestClassifier
rf=RandomForestClassifier()
rf.fit(x_train,y_train)

#Predict values for cv data
pred_cv2=rf.predict(x_cv)

#Evaluate accuracy of model
accuracy_score(y_cv,pred_cv2) #92.39%
matrix2=confusion_matrix(y_cv,pred_cv2)
print(matrix2)

#(d)SUPPORT VECTOR MACHINE (SVM) ALGORITHM
from sklearn import svm
svm_model=svm.SVC()
svm_model.fit(x_train,y_train)

#Predict values for cv data
pred_cv3=svm_model.predict(x_cv)

#Evaluate accuracy of model
accuracy_score(y_cv,pred_cv3) #61.96%
matrix3=confusion_matrix(y_cv,pred_cv3)
print(matrix3)

#(e)NAIVE BAYES ALGORITHM
from sklearn.naive_bayes import GaussianNB 
nb=GaussianNB()
nb.fit(x_train,y_train)

#Predict values for cv data
pred_cv4=nb.predict(x_cv)

#Evaluate accuracy of model
accuracy_score(y_cv,pred_cv4) #88%
matrix4=confusion_matrix(y_cv,pred_cv4)
print(matrix4)


#(g) GRADIENT BOOSTING MACHINE ALGORITHM
from sklearn.ensemble import GradientBoostingClassifier

gbm=GradientBoostingClassifier()

gbm.fit(x_train,y_train)

#Predict values for cv data
pred_cv6=gbm.predict(x_cv)

#Evaluate accuracy of model
accuracy_score(y_cv,pred_cv6) #93.48%
matrix6=confusion_matrix(y_cv,pred_cv6)
print(matrix6)

#Select best model in order of accuracy
#Naive Bayes - 88%
#Logistic Regression - 92.39%
#Gradient Boosting Machine -93.48%
#Random Forest - 92.39%
#Decision Tree - 86.96%
#Support Vector Machine - 61.96%


#Predict values using test data (Decision tree algorithm- as this has high accuracy)
pred_test=gbm.predict(x_cv)
#Write test results in csv file
predictions=pd.DataFrame(pred_test, 
   columns=['Predictions']).to_excel('Fraud_Predictions.xls')
