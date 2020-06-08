


import numpy as np # linear algebra
import pandas as pd # data processing

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split,KFold,cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
import matplotlib.pyplot as plt

data=pd.read_csv(r'C:\\Users\\Siddharth\\Desktop\\Misc\\Kaggle Datasets\\heart_disease_prediction_dataset.csv')
print("Printing the first 5 rows of the DataFrame : ")
print(data.head())
print("\n")

print("printing/getting the names of all columns in our dataframe as follows : ")
print(data.columns)
print("\n")

##  # Checks for null Values, Returns Boolean Array
print("now we check for null values in our dataframe : ")
print(data.isnull())
print("\n")

print("getting the sum of all null values in our dataframe as : ")
print(data.isnull().sum())
print("\n")

print("the number of rows and columns in pur dataframe is as follows : ",data.shape)
print("\n")

print("the size of the dataframe is : ",data.size)
print("\n")

print("If we want to see what all the data types are in our dataframe : ")
print(data.dtypes)
print("\n")

print(" Index, Datatype and Memory information of our dataframe is as follows : ")
print(data.info())
print("\n")

print(" Summary statistics for all numerical predictors/columns is as follows : ")
print(data.describe())
print("\n")

## dropping the null values in our dataset
data=data.dropna()
print(data.isnull().sum())
print("\n")

print("the correlation between columns/predictors is as follows : ")
print(data.corr())
print("\n")

print(data.groupby('TenYearCHD').sum())

print(data.groupby('TenYearCHD').size())

print("the max of all columns is as follows : ")
print(data.max())
print("\n")

print("the min of all columns is as follows : ")
print(data.min())
print("\n")

print("the mean of all columns is as follows : ")
print(data.mean())
print("\n")

print("the median of all predictor columns is as follows : ")
print(data.median())
print("\n")

print("the standard deviation of all predictor columns is as follows : ")
print(data.std())
print("\n")

X=data[data.columns]
X=X.drop(columns=['TenYearCHD'])
print(X.head())

Y=data['TenYearCHD']
print(Y.head())

from sklearn.feature_selection import SelectKBest,chi2
test=SelectKBest(score_func=chi2,k=10)
fit=test.fit(X,Y)
print(fit.scores_)
print("\n")

print(X.columns)
print("\n")

X=data[['age','cigsPerDay','totChol','sysBP','diaBP','glucose']]
print(X.head())
print("\n")

from sklearn.preprocessing import StandardScaler
sc_x=StandardScaler()
X=sc_x.fit_transform(X)

#histograms the dataset
fig = plt.figure(figsize = (15,20))
ax = fig.gca()
pd.DataFrame(X).hist(ax = ax)

models=[]
models.append(('LR',LogisticRegression()))
models.append(('DT',DecisionTreeClassifier()))
models.append(('KN',KNeighborsClassifier()))
models.append(('NB',GaussianNB()))
models.append(('SVC',SVC()))

from sklearn.metrics import accuracy_score

results=[]
names=[]
scoring='accuracy'
for name,model in models:
    kfold=KFold(n_splits=10,random_state=7)
    cv_result=cross_val_score(model,X,Y,cv=kfold,scoring=scoring)
    results.append(cv_result)
    names.append(name)
    msg=("%s: %f (%f)" % (name,cv_result.mean(),cv_result.std()))
    print(msg)

import matplotlib.pyplot as plt
fig=plt.figure()
fig.suptitle('Algorithms Coparison')
ax=fig.add_subplot(111)
plt.boxplot(results)
ax.set_xticklabels(names)
plt.show()

x_train,x_test,y_train,y_test=train_test_split(X,Y,test_size=0.2,random_state=1)

my_model=LogisticRegression()
my_model.fit(x_train,y_train)

result=my_model.score(x_test,y_test)
print('Accuracy : ' ,(result*100))

from sklearn.metrics import confusion_matrix

predicted = my_model.predict(x_test)
matrix = confusion_matrix(y_test, predicted)
print(matrix)



