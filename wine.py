# -*- coding: utf-8 -*-
"""
Created on Wed Jan 19 21:37:30 2022

@author: cheng164
"""

import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt
import seaborn as sns
from imblearn.over_sampling import SMOTE

from sklearn import preprocessing
from sklearn import svm
from sklearn import tree

from sklearn.metrics import accuracy_score, roc_auc_score, classification_report, plot_roc_curve
from sklearn.metrics import f1_score
from sklearn.metrics import confusion_matrix
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.neighbors import KNeighborsClassifier

from sklearn.model_selection import validation_curve
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import learning_curve
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline, Pipeline

from utilities import tree_accuracy_vs_alpha, validation_curve_plot, GridSearchCV_result, learning_curve_plot, loss_curve_plot
from sklearn.exceptions import ConvergenceWarning
from warnings import simplefilter


## Data Loading and Visualization
df=pd.read_csv("winequalityN.csv")
df.head()

##Describing the data
df.describe()
plt.figure()
sns.countplot(df['type'])
plt.figure()
sns.countplot(df['quality'])
df['quality'].value_counts()

## Mapping values of target variable quality to 'low', 'medium' and 'high' categories for classification
df['quality']=df['quality'].map({3:'low', 4:'low', 5:'medium', 6:'medium', 7:'medium', 8:'high', 9:'high'})
df['quality']=df['quality'].map({'low':0,'medium':1, 'high':2})
df.sample(5)


# df['quality']=df['quality'].map({3:'low', 4:'low', 5:'low', 6:'high', 7:'high', 8:'high', 9:'high'})
# df['quality']=df['quality'].map({'low':0,'high':1})
# df.sample(5)


##Checking for missing values as per column
df.isna().sum()

##Fill the missing values
for col, value in df.items():
    if col != 'type':
        df[col] = df[col].fillna(df[col].mean())

# df=df.fillna(df.mean())
# df.isnull().values.any()


## Removal of Outliers
#plotting a boxplot to view the general distribution of data across all features to check for outliers.
sns.set()
plt.figure(figsize=(30,15))
sns.boxplot(data=df)
plt.show()


fig, ax =plt.subplots(1,3)
plt.subplots_adjust(right=2.5, top=1.5)
sns.boxplot(df['residual sugar'], df['type'], ax=ax[0])
sns.boxplot(df['free sulfur dioxide'], df['type'], ax=ax[1])
sns.boxplot(df['total sulfur dioxide'], df['type'], ax=ax[2])
plt.show()


#Removing outliers in residual sugar
lower = df['residual sugar'].mean()-3*df['residual sugar'].std()
upper = df['residual sugar'].mean()+3*df['residual sugar'].std()
df = df[(df['residual sugar']>lower) & (df['residual sugar']<upper)]

#Removing outliers in free sulfur dioxide
lower = df['free sulfur dioxide'].mean()-3*df['free sulfur dioxide'].std()
upper = df['free sulfur dioxide'].mean()+3*df['free sulfur dioxide'].std()
df = df[(df['free sulfur dioxide']>lower) & (df['free sulfur dioxide']<upper)]

#Removing outliers in total sulfur dioxide
lower = df['total sulfur dioxide'].mean()-3*df['total sulfur dioxide'].std()
upper = df['total sulfur dioxide'].mean()+3*df['total sulfur dioxide'].std()
df = df[(df['total sulfur dioxide']>lower) & (df['total sulfur dioxide']<upper)]


##encoding the wine type attribute
le = preprocessing.LabelEncoder()
df['type'] = le.fit_transform(df['type'])
df.head()


##Explore correlations visually
corr = df.corr()
plt.subplots(figsize=(10, 10))
sns.heatmap(corr, annot=True,cmap=sns.diverging_palette(200, 10, as_cmap=True), vmin=-1, vmax=1, linewidths=.5, fmt=".2f")

#%%  Data preprossesing
 
X = df.drop(['quality'], axis=1)
y = df['quality']
 
print("Splitting into train/test sets...")
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0, shuffle=True, stratify=y)

cv=StratifiedKFold(n_splits=5, shuffle=False, random_state=None)

#################################   not suitable for preprocessing within cross validation fold
# scaler = preprocessing.StandardScaler()
# X_train = scaler.fit_transform(X_train)
# X_test = scaler.transform(X_test)
####################################


## Oversampling to have balanced data if necessary
# oversampled = SMOTE(random_state=0)
# X_train, y_train = oversampled.fit_resample(X_train, y_train)
# y_train.value_counts()


## Define scoring metric depending on it's binary or multiclass classification problem
if y_test.nunique()>2:   # multiclass case
    scoring_metric = 'f1_macro' 
    classes = ['Low', 'Medium', 'High']
else:
    scoring_metric = 'accuracy' 
    classes = ['Low', 'High']
    
#%% Decision Tree

print("Starts to fit Decision Tree...")

## Accuracy vs alpha for training and testing sets
tree_accuracy_vs_alpha(X_train, y_train, X_test, y_test)

clf = tree.DecisionTreeClassifier(random_state=0)
pipeline = Pipeline([('scaler', preprocessing.StandardScaler()), ('clf', clf)])

## validation curve
validation_curve_plot(pipeline, X_train, y_train, clf_name = "Decision Tree", param_name= 'clf__max_depth', param_range=np.arange(1,40,4), scoring= scoring_metric, cv=cv)
validation_curve_plot(pipeline, X_train, y_train, clf_name = "Decision Tree", param_name= 'clf__ccp_alpha', param_range=np.linspace(0,0.001,5), scoring= scoring_metric, cv=cv)
validation_curve_plot(pipeline, X_train, y_train, clf_name = "Decision Tree", param_name= 'clf__min_samples_leaf', param_range= np.arange(1,5), scoring= scoring_metric, cv=cv)

## GridSearchCV
param_grid = {'clf__max_depth': np.arange(3,15,2), 'clf__min_samples_split': [2,3,4], 'clf__min_samples_leaf': [1,2]} ## Pre Pruning
# param_grid = {'clf__max_depth': np.arange(3,15,2), 'clf__ccp_alpha': np.linspace(0,0.002,5)}  ## Post Pruning
gscv_tree, _ = GridSearchCV_result(pipeline, X_train, y_train, X_test, y_test, param_grid=param_grid, scoring= scoring_metric, cv=cv, classes = classes)

## Learning Curve
train_scores, test_scores = learning_curve_plot(gscv_tree, "Decision Tree", X_train, y_train, train_size_pct= np.linspace(0.2,1.0,5), scoring= scoring_metric, cv=cv)

#%% KNN

print("Starts to fit KNN...")

clf = KNeighborsClassifier()
pipeline = Pipeline([('scaler', preprocessing.StandardScaler()), ('clf', clf)])

validation_curve_plot(pipeline, X_train, y_train, clf_name = "KNN", param_name= 'clf__n_neighbors', param_range=np.arange(1,40,2), scoring= scoring_metric, cv=cv)
validation_curve_plot(pipeline, X_train, y_train, clf_name = "KNN", param_name= 'clf__metric', param_range= ['euclidean', 'manhattan', 'minkowski'], scoring= scoring_metric, cv=cv)
validation_curve_plot(pipeline, X_train, y_train, clf_name = "KNN", param_name= 'clf__p', param_range= [1, 2, 3, 4], scoring= scoring_metric, cv=cv)

## GridSearchCV
param_grid = {'clf__n_neighbors': np.arange(3,20,2)}
gscv_tree, _ = GridSearchCV_result(pipeline, X_train, y_train, X_test, y_test, param_grid=param_grid, scoring= scoring_metric, cv=cv, classes = classes)

## Learning Curve
train_scores, test_scores = learning_curve_plot(gscv_tree, "KNN", X_train, y_train, train_size_pct= np.linspace(0.2,1.0,5), scoring= scoring_metric, cv=cv)



#%% SVM

print("Starts to fit SVM...")


## validation curve
#clf = svm.SVC(kernel='poly', random_state=0)
#pipeline = Pipeline([('scaler', preprocessing.StandardScaler()), ('clf', clf)])
#validation_curve_plot(pipeline, X_train, y_train, clf_name = "SVM", param_name= 'clf__degree', param_range=[2,3,4,5,6], scoring=scoring_metric, cv=cv)
#validation_curve_plot(pipeline, X_train, y_train, clf_name = "SVM", param_name= 'clf__C', param_range=np.logspace(-3,3,10), scoring=scoring_metric, cv=cv)
#validation_curve_plot(pipeline, X_train, y_train, clf_name = "SVM", param_name= 'clf__gamma', param_range=np.logspace(-3,3,10), scoring=scoring_metric, cv=cv)

#clf = svm.SVC(kernel='rbf', random_state=0)
#pipeline = Pipeline([('scaler', preprocessing.StandardScaler()), ('clf', clf)])
#validation_curve_plot(pipeline, X_train, y_train, clf_name = "SVM", param_name= 'clf__C', param_range=np.logspace(-3,3,10), scoring=scoring_metric, cv=cv)
#validation_curve_plot(pipeline, X_train, y_train, clf_name = "SVM", param_name= 'clf__gamma', param_range=np.logspace(-3,3,10), scoring=scoring_metric, cv=cv)


# GridSearchCV
#kernel_functions = ['linear','poly','rbf']
kernel_functions = ['rbf']

gscv_tree_list = []
gscv_best_scores = []

for kernel in kernel_functions:  
    
    clf = svm.SVC(kernel= kernel, probability=True, random_state=0)
    pipeline = Pipeline([('scaler', preprocessing.StandardScaler()), ('clf', clf)])
    
    if kernel == 'linear':
        print("Working on linear model......")
        param_grid = {'clf__C': np.logspace(-3,2,5)}
        gscv_clf, best_score = GridSearchCV_result(pipeline, X_train, y_train, X_test, y_test, param_grid=param_grid, scoring=scoring_metric, cv=cv)

    elif kernel == 'poly':
        print("Working on poly model......")
        param_grid = { 'clf__degree': [2,3,4,5]}
        gscv_clf, best_score = GridSearchCV_result(pipeline, X_train, y_train, X_test, y_test, param_grid=param_grid, scoring=scoring_metric, cv=cv)
        
    else:
        print("Working on RBF model......")
        param_grid = {'clf__C': np.logspace(-3,2,5), 'clf__gamma': np.logspace(-3,2,5)}    
        gscv_clf, best_score = GridSearchCV_result(pipeline, X_train, y_train, X_test, y_test, param_grid=param_grid, scoring=scoring_metric, cv=cv)
    
    gscv_tree_list.append(gscv_clf)
    gscv_best_scores.append(best_score)


max_score_index = gscv_best_scores.index(max(gscv_best_scores))
best_gscv_clf =  gscv_tree_list[max_score_index]  
print('Best SVM model is {}. Best score is {}. Best hyperparameters are : {}'.format(kernel_functions[max_score_index], best_gscv_clf.best_score_, best_gscv_clf.best_params_ ))
    

# clf = svm.SVC(probability=True, random_state=0)
# pipeline = Pipeline([('scaler', preprocessing.StandardScaler()), ('clf', clf)])
# param_grid = {'clf__kernel':['poly','rbf'], 'clf__C': np.logspace(-3,2,4), 'clf__degree': [2,3,4], 'clf__gamma': np.logspace(-3,2,4)}
# gscv_clf, _ = GridSearchCV_result(pipeline, X_train, y_train, X_test, y_test, param_grid=param_grid, scoring= scoring_metric, cv=cv) 

## Learning Curve
train_scores, test_scores = learning_curve_plot(best_gscv_clf, "SVM", X_train, y_train, train_size_pct= np.linspace(0.2,1.0,5), scoring= scoring_metric, cv=cv)

#%% Boosting

print("Starts to fit Boosting...")

clf = AdaBoostClassifier(base_estimator=tree.DecisionTreeClassifier(max_depth=5, random_state=0))
pipeline = Pipeline([('scaler', preprocessing.StandardScaler()), ('clf', clf)])
## validation curve
validation_curve_plot(pipeline, X_train, y_train, clf_name = "Boosting", param_name="clf__n_estimators", param_range=np.arange(5,100,5), scoring= scoring_metric, cv=cv)
validation_curve_plot(pipeline, X_train, y_train, clf_name = "Boosting", param_name="clf__learning_rate", param_range=np.logspace(-4,1,4), scoring= scoring_metric, cv=cv)

## GridSearchCV
param_grid = {'clf__n_estimators':np.arange(5,50,5), 'clf__learning_rate': np.logspace(-4,0,6)}
gscv_tree, _ = GridSearchCV_result(pipeline, X_train, y_train, X_test, y_test, param_grid=param_grid, scoring= scoring_metric, cv=cv, classes = classes)

## Learning Curve
learning_curve_plot(gscv_tree, "Boosting", X_train, y_train, train_size_pct= np.linspace(0.2,1.0,5), scoring= scoring_metric, cv=cv)

#%% ANN

simplefilter("ignore", category=ConvergenceWarning)  ## Disable the not converge warning message

print("Starts to fit NN...")

## validation curve
clf = MLPClassifier(random_state=0, max_iter=1000, activation = 'relu')
pipeline = Pipeline([('scaler', preprocessing.StandardScaler()), ('clf', clf)])
validation_curve_plot(pipeline, X_train, y_train, clf_name = 'NN', param_name="clf__hidden_layer_sizes", param_range= [20,40,60,80,100], scoring= scoring_metric, cv=cv)

clf = MLPClassifier(hidden_layer_sizes= (100,),  random_state=0, max_iter=1000, activation = 'relu', early_stopping = False)
pipeline = Pipeline([('scaler', preprocessing.StandardScaler()), ('clf', clf)])
validation_curve_plot(pipeline, X_train, y_train, clf_name = 'NN', param_name="clf__learning_rate_init", param_range=np.logspace(-5,1,10), scoring= scoring_metric, cv=cv)
validation_curve_plot(pipeline, X_train, y_train, clf_name = 'NN', param_name="clf__alpha", param_range=np.logspace(-5,2,10), scoring= scoring_metric, cv=cv)

## GridSearchCV

param_grid = {'clf__learning_rate_init':[1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1], 'clf__alpha': np.logspace(-4,1,6)}
gscv_tree, gscv_score = GridSearchCV_result(pipeline, X_train, y_train, X_test, y_test, param_grid=param_grid, scoring= scoring_metric, cv=cv)

## Learning Curve
train_scores, test_scores = learning_curve_plot(gscv_tree, "NN", X_train, y_train, train_size_pct= np.linspace(0.2,1.0,5), scoring= scoring_metric, cv=cv)
loss_curve_plot(gscv_tree.best_estimator_['clf'])