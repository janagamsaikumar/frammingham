import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt
import seaborn as sns 

dataset=pd.read_csv(r'C:\Users\saikumar\Desktop\AMXWAM data science\AMXWAM_ TASK\TASK-24\framingham.csv')
# this dataset contains the medical details of male persons
# based on those details we have to find that how many of them will be having heart attack or not
dataset.columns
dataset.isnull().sum() # checking our dataset having null values or not 

# splitting our dataset into dependent and independent variables
X=dataset.iloc[:,:15].values
y=dataset.iloc[:,-1].values
# we have to treat missing values
from sklearn.impute import SimpleImputer
imputer=SimpleImputer(missing_values=np.nan,strategy='mean',)
X=imputer.fit_transform(X)

# there are some values with high magnitude we have to reduce it to 0 to 1 to make machine understand
# we have to standardize X using feature scaling  
from sklearn.preprocessing import StandardScaler
sc=StandardScaler()
X=sc.fit_transform(X)

# now its to split our data into train and test
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=1)


# now predict 
from sklearn.linear_model import LogisticRegression
reg=LogisticRegression()
reg.fit(X_train,y_train)

y_pred=reg.predict(X_test)

from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test,y_pred)
print(cm)

# find the accuracy and come to conclusion
from sklearn.metrics import accuracy_score
acc=accuracy_score(y_test,y_pred)
print('accuracy is',acc)



