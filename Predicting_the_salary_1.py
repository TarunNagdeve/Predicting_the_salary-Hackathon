import pandas as pd
import  numpy as np
import re
train=pd.read_csv(r'C:\Users\TARUN\Desktop\New folder (3)\Predicting_The_Salary\Data\Final_Train_Dataset.csv')
train['salary']=train['salary'].map({'6to10':1, '10to15':2, '15to25':3, '3to6':4, '25to50':5, '0to3':6})
Labels=train['salary']
test=pd.read_csv(r'C:\Users\TARUN\Desktop\New folder (3)\Predicting_The_Salary\Data\Final_Test_Dataset.csv')
train['job_type']=train['job_type'].map({'Analytics':2, 'analytics':3, 'Analytic':4, 'ANALYTICS':5, 'analytic':6})
train['job_type']=train['job_type'].fillna(1)
job=[]
for i in range(1,1505):
    job.append(i)
###################################################################################################################
loc=[]
for i in train['location'].unique():
    loc.append(i)
for i,j in zip(train['location'].unique(),job):
    train['location']=train['location'].replace(i,j)
for i,j in zip(loc,job):
    test['location'] = test['location'].replace(i, j)
pattern=r'[A-Za-z]+'
lol=[]
for i in (test['location'].unique()):
    if(re.search(pattern,str(i))):
        lol.append(i)
job2=[]
for i in range(1505,1720):
    job2.append(i)
for i,j in zip(lol,job2):
    test['location'] = test['location'].replace(i, j)
#######################################################################################################################
skills=[]
for i in range(1,12953):
    skills.append(i)
my_skills=[]
for i in train['key_skills'].unique():
    my_skills.append(i)

for i,j in zip(train['key_skills'].unique(),skills):
    train['key_skills']=train['key_skills'].replace(i,j)

for i,j in zip(my_skills,skills):
    test['key_skills']=test['key_skills'].replace(i,j)
lol2=[]
for i in (test['key_skills'].unique()):
    if (re.search(pattern, str(i))):
        lol2.append(i)

job3=[]
for i in range(12953,16000):
    job3.append(i)
for i,j in zip(lol2,job3):
    test['key_skills'] = test['key_skills'].replace(i, j)
#####################################################################################################
designation=[]
for i in train['job_desig'].unique():
    designation.append(i)
count_desig=[]
for i in range(1,11709):
    count_desig.append(i)

for i,j in zip(designation,count_desig):
    train['job_desig']=train['job_desig'].replace(i,j)

for i,j in zip(designation,count_desig):
    test['job_desig']=test['job_desig'].replace(i,j)

lol3=[]
for i in (test['job_desig'].unique()):
    if (re.search(pattern, str(i))):
        lol3.append(i)
job4=[]
for i in range(11709,14000):
    job4.append(i)

for i,j in zip(lol3,job4):
    test['job_desig']=test['job_desig'].replace(i,j)

#####################################################################################################
description=[]

for i in (train['job_description'].unique()):
    description.append(i)
description_count=[]
for i in range(1,9061):
    description_count.append(i)
for i,j in zip(description,description_count):
    train['job_description']=train['job_description'].replace(i,j)

for i,j in zip(description,description_count):
    test['job_description']=test['job_description'].replace(i,j)
lol6=[]
for i in test['job_description'].unique():
    if (re.search(pattern, str(i))):
        lol6.append(i)
job6=[]
for i in range(9061,11000):
    job6.append(i)

for i,j in zip(lol6,job6):
    test['job_description'] = test['job_description'].replace(i, j)
##################################################################################################################
experience=[]
for i in train['experience'].unique():
    experience.append(i)
experience_count=[]
for i in range(1,130):
    experience_count.append(i)
for i,j in zip(experience,experience_count):
    train['experience']=train['experience'].replace(i,j)

for i,j in zip(experience,experience_count):
    test['experience']=test['experience'].replace(i,j)

test['experience']=test['experience'].replace('15-19 yrs',130)
test['experience']=test['experience'].replace('11-19 yrs',131)
test['experience']=test['experience'].replace('25-30 yrs',132)
############################################################
df1=pd.DataFrame(test)
df1.to_csv(r'C:\Users\TARUN\Desktop\New folder (3)\trunn.csv')
#######################################################################3







#########################################
test=pd.read_csv(r'C:\Users\TARUN\Desktop\New folder (3)\Predicting_The_Salary\Data\trunn.csv')
test['job_type']=test['job_type'].map({'Analytics':2, 'analytics':3, 'Analytic':4, 'ANALYTICS':5, 'analytic':6})
test['job_type']=test['job_type'].fillna(1)
test_feature=Features=[test['experience'],test['job_description'],test['job_desig'],test['job_type'],test['key_skills'],test['location'],test['company_name_encoded']]
test_feature=np.array(test_feature).reshape(6601,7)
ans=[]
for i in dmodel.predict(test_feature):
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

df=pd.DataFrame(ans)
df.to_csv(r'C:\Users\TARUN\Desktop\New folder (3)\Predicting_The_Salary\Data\my_salary_predictor.csv')
