#!/usr/bin/env python
# coding: utf-8

# In[15]:


#fitting and predicting
from sklearn.ensemble import RandomForestClassifier
clf = RandomForestClassifier(random_state=0)
X = [[ 1,  2,  3],  # 2 samples, 3 features
     [11, 12, 13]]
y = [0, 1]  # classes of each sample
clf.fit(X, y)

clf.predict(X) #predict classes of X
clf.predict([[4,5,6],[14,15,16]])#predict new values 

#transformers and preprocessors
from sklearn.preprocessing import StandardScaler
X = [[0, 15],
     [1, -10]]
StandardScaler().fit(X).transform(X)

#pipelines. transformers and estimators
from pandas import read_csv
from pandas.plotting import scatter_matrix
from matplotlib import pyplot
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.datasets import load_iris

pipe=make_pipeline(StandardScaler(),LogisticRegression(random_state=0))#creating pipeline object


X,y=load_iris(return_X_y=True)
X_train,X_test,y_train,y_test=train_test_split(X,y,random_state=0)#load iris dataset and split it into train and test sets

pipe.fit(X_train,y_train)#fit the whole pipeline

accuracy_score(pipe.predict(X_test),y_test)#accurancy


# In[19]:


#model evaluation
from sklearn.datasets import make_regression
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import cross_validate
X,y=make_regression(n_samples=1000,random_state=0)
lr=LinearRegression()

result=cross_validate(lr,X,y) #defaults to 5-fold
result  # r_squared score is high because dataset is easy


# In[24]:


#Automatic parameter search
from sklearn.datasets import fetch_california_housing
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import RandomizedSearchCV
from scipy.stats import randint

X, y = fetch_california_housing(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)

# define the parameter space that will be searched over
param_distributions = {'n_estimators': randint(1, 5),
                       'max_depth': randint(5, 10)}

# now create a searchCV object and fit it to the data
search = RandomizedSearchCV(estimator=RandomForestRegressor(random_state=0),
                            n_iter=5,
                            param_distributions=param_distributions,
                            random_state=0)

search.fit(X_train,y_train)
search.fit(X_train, y_train)
search.best_params_
search.score(X_test,y_test)


# In[6]:





# In[ ]:




