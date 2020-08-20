import pandas as pd
import  numpy as np
import re
train=pd.read_csv(r'C:\Users\TARUN\Desktop\New folder (3)\Predicting_The_Salary\Data\lets_predict.csv')
Labels=train['salary']
train['extra']=train['experience']*train['job_description']
Features=[train['experience'],train['job_description'],train['job_desig'],train['key_skills'],train['location'],train['company_name_encoded'],train['job_type']]

Features=(np.array(Features).reshape(19802,7))
Labels=(np.array(Labels).reshape(19802,1))
from sklearn.preprocessing import StandardScaler
sc=StandardScaler()
scaled_Features=sc.fit_transform(Features)
from sklearn.preprocessing import PolynomialFeatures
poly=PolynomialFeatures(3)
x_poly=poly.fit_transform(scaled_Features)
from sklearn.decomposition import PCA
pca=PCA(n_components=3)
x_pca=pca.fit_transform(x_poly)
from sklearn.model_selection import train_test_split
xtrain,xtest,ytrain,ytest=train_test_split(x_pca,Labels,test_size=0.2)
from sklearn.metrics import  classification_report
from sklearn.svm import SVC
smodel=SVC()
smodel.fit(xtrain,ytrain)
print(smodel.score(xtest,ytest))

test=pd.read_csv(r'C:\Users\TARUN\Desktop\New folder (3)\trunn.csv')
test['job_type']=test['job_type'].map({'Analytics':2, 'analytics':3, 'Analytic':4, 'ANALYTICS':5, 'analytic':6})
test['job_type']=test['job_type'].fillna(1)
test_features=[test['experience'],test['job_description'],test['job_desig'],test['key_skills'],test['location'],test['company_name_encoded'],test['job_type']]
test_features=(np.array(test_features).reshape(6601,7))
scaled_test_features=sc.transform(test_features)

smodel.fit(scaled_Features,Labels)
ans=[]
for i in smodel.predict(scaled_test_features):
    if(i==1):
        ans.append('6to10')
    elif(i==2):
        ans.append('10to15')
    elif(i==3):
        ans.append('15to25')
    elif (i == 4):
        ans.append('3to6')
    elif (i == 5):
        ans.append('25to50')
    elif (i == 6):
        ans.append('0to3')

