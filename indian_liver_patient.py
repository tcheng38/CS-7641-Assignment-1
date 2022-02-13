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
from  warnings import simplefilter


## Data Loading and Visualization
df=pd.read_csv("indian_liver_patient.csv")
df.head()

#describing the data
print(df.info())
df.describe()

#encoding the Gender attribute
plt.figure()
sns.countplot(df.Gender)
df['Gender'].replace({'Male':1,'Female':0},inplace=True)
df['Dataset'].replace(2,0, inplace=True)
# let's look on target variable - classes imbalanced?
df['Dataset'].value_counts()
plt.figure()
sns.countplot(df.Dataset)


#checking for missing values as per column
df.isna().sum()

#checking the rows with the missing values
df[df['Albumin_and_Globulin_Ratio'].isna()]
df["Albumin_and_Globulin_Ratio"] = df.Albumin_and_Globulin_Ratio.fillna(df['Albumin_and_Globulin_Ratio'].mean())

# Explore correlations visually
f, ax = plt.subplots(figsize=(12,6))
sns.heatmap(df.corr(), annot=True, fmt='.2f')

#%%  Data preprossesing
 
X = df.drop(['Dataset'], axis=1)
y = df['Dataset']
 
print("Splitting into train/test sets...")
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0, shuffle=True, stratify=y)

cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=None)

#################################   not suitable for preprocessing within cross validation fold
# scaler = preprocessing.StandardScaler()
# X_train = scaler.fit_transform(X_train)
# X_test = scaler.transform(X_test)
####################################

#######################################
## Oversampling to have balanced data
#oversampled = SMOTE(random_state=0)
#X_train, y_train = oversampled.fit_resample(X_train, y_train)
#y_train.value_counts()
########################################

## Define scoring metric depending on it's binary or multiclass classification problem
if y_test.nunique()>2:   # multiclass case
    scoring_metric = 'f1_macro' 
else:
    scoring_metric = 'balanced_accuracy' 

#%% Decision Tree

print("Starts to fit Decision Tree...")

## To evaluate the entire pipeline of data preparation and model together as a single atomic unit.
clf = tree.DecisionTreeClassifier(random_state=0)
pipeline = Pipeline([('scaler', preprocessing.StandardScaler()), ('clf', clf)])

## Accuracy vs alpha for training and testing sets
tree_accuracy_vs_alpha(X_train, y_train, X_test, y_test)

## validation curve
validation_curve_plot(pipeline, X_train, y_train, clf_name = "Decision Tree", param_name= 'clf__max_depth', param_range=np.arange(1,15,1), scoring= scoring_metric, cv=cv)
validation_curve_plot(pipeline, X_train, y_train, clf_name = "Decision Tree", param_name= 'clf__ccp_alpha', param_range=np.linspace(0,0.03,10), scoring= scoring_metric, cv=cv)
validation_curve_plot(pipeline, X_train, y_train, clf_name = "Decision Tree", param_name= 'clf__min_samples_leaf', param_range= np.arange(1,5), scoring= scoring_metric, cv=cv)

## GridSearchCV 
param_grid = {'clf__max_depth': np.arange(3,10), 'clf__min_samples_split': [2,3,4], 'clf__min_samples_leaf': [1,2]} ## Pre Pruning
# param_grid = {'clf__max_depth': np.arange(4,10), 'clf__ccp_alpha': np.linspace(0.01,0.04,5)}  ## Post Pruning
gscv_tree, _ = GridSearchCV_result(pipeline, X_train, y_train, X_test, y_test, param_grid=param_grid, scoring= scoring_metric, cv=cv)

## Learning Curve
learning_curve_plot(gscv_tree, "Decision Tree", X_train, y_train, train_size_pct= np.linspace(0.2,1.0,5), scoring= scoring_metric, cv=cv)

## Plot tree
clf = gscv_tree.best_estimator_['clf'].fit(X_train, y_train)
tree.plot_tree(clf)
plt.show()


## Learning Curve
learning_curve_plot(gscv_tree, "Decision Tree", X_train, y_train, train_size_pct= np.linspace(0.2,1.0,5), scoring= scoring_metric, cv=5)
#####################################################################
#%% KNN

print("Starts to fit KNN...")

clf = KNeighborsClassifier()
pipeline = Pipeline([('scaler', preprocessing.StandardScaler()), ('clf', clf)])

## validation curve
validation_curve_plot(pipeline, X_train, y_train, clf_name = "KNN", param_name= 'clf__n_neighbors', param_range=np.arange(1,40,2), scoring= scoring_metric, cv=cv)
validation_curve_plot(pipeline, X_train, y_train, clf_name = "KNN", param_name= 'clf__metric', param_range= ['euclidean', 'manhattan', 'minkowski'], scoring= scoring_metric, cv=cv)
validation_curve_plot(pipeline, X_train, y_train, clf_name = "KNN", param_name= 'clf__p', param_range= [1, 2, 3, 4], scoring= scoring_metric, cv=cv)


## GridSearchCV
param_grid = {'clf__n_neighbors':np.arange(1, 45, 2), 'clf__p':[1, 2, 3, 4]}
gscv_tree, _ = GridSearchCV_result(pipeline, X_train, y_train, X_test, y_test, param_grid=param_grid, scoring= scoring_metric, cv=cv)

## Learning Curve
train_scores, test_scores = learning_curve_plot(gscv_tree, "KNN", X_train, y_train, train_size_pct= np.linspace(0.2,1.0,5), scoring= scoring_metric, cv=cv)

#%% SVM

print("Starts to fit SVM...")

## validation curve
clf = svm.SVC(kernel='poly', random_state=0)
pipeline = Pipeline([('scaler', preprocessing.StandardScaler()), ('clf', clf)])
validation_curve_plot(pipeline, X_train, y_train, clf_name = "SVM", param_name= 'clf__degree', param_range=[2,3,4,5,6], scoring=scoring_metric, cv=cv)

clf = svm.SVC(kernel='rbf', random_state=0)
pipeline = Pipeline([('scaler', preprocessing.StandardScaler()), ('clf', clf)])
validation_curve_plot(pipeline, X_train, y_train, clf_name = "SVM", param_name= 'clf__C', param_range=np.logspace(-3,3,10), scoring=scoring_metric, cv=cv)
validation_curve_plot(pipeline, X_train, y_train, clf_name = "SVM", param_name= 'clf__gamma', param_range=np.logspace(-3,3,10), scoring=scoring_metric, cv=cv)


# GridSearchCV
kernel_functions = ['linear','poly','rbf']
gscv_tree_list = []
gscv_best_scores = []

for kernel in kernel_functions:  
    
    clf = svm.SVC(kernel= kernel, probability=True, random_state=0)
    pipeline = Pipeline([('scaler', preprocessing.StandardScaler()), ('clf', clf)])
    
    if kernel == 'linear':
        param_grid = {'clf__C': np.logspace(-3,2,5)}
        gscv_clf, best_score = GridSearchCV_result(pipeline, X_train, y_train, X_test, y_test, param_grid=param_grid, scoring=scoring_metric, cv=cv)

    elif kernel == 'poly':
        param_grid = {'clf__C': np.logspace(-3,3,4), 'clf__degree': [2,3,4]}
        gscv_clf, best_score = GridSearchCV_result(pipeline, X_train, y_train, X_test, y_test, param_grid=param_grid, scoring=scoring_metric, cv=cv)
        
    else:
        param_grid = {'clf__C': np.logspace(-4,3,8), 'clf__gamma': np.logspace(-4,3,8)}    
        gscv_clf, best_score = GridSearchCV_result(pipeline, X_train, y_train, X_test, y_test, param_grid=param_grid, scoring=scoring_metric, cv=cv)
    
    gscv_tree_list.append(gscv_clf)
    gscv_best_scores.append(best_score)


max_score_index = gscv_best_scores.index(max(gscv_best_scores))
best_gscv_clf =  gscv_tree_list[max_score_index]  
print('Best SVM model is {}. Best score is {}. Best hyperparameters are : {}'.format(kernel_functions[max_score_index], best_gscv_clf.best_score_, best_gscv_clf.best_params_ ))
    
#######################################################
# clf = svm.SVC(probability=True, random_state=0)
# pipeline = Pipeline([('scaler', preprocessing.StandardScaler()), ('clf', clf)])
# param_grid = {'clf__kernel':['poly','rbf'], 'clf__C': np.logspace(-3,2,4), 'clf__degree': [2,3,4], 'clf__gamma': np.logspace(-3,2,4)}
# gscv_clf, _ = GridSearchCV_result(pipeline, X_train, y_train, X_test, y_test, param_grid=param_grid, scoring= scoring_metric, cv=cv) 
###########################################################

## Learning Curve
train_scores, test_scores = learning_curve_plot(best_gscv_clf, "SVM", X_train, y_train, train_size_pct= np.linspace(0.2,1.0,5), scoring= scoring_metric, cv=cv)

#%% Boosting

print("Starts to fit Boosting...")

clf = AdaBoostClassifier(base_estimator=tree.DecisionTreeClassifier(max_depth=2, min_samples_leaf=1), random_state=0)
pipeline = Pipeline([('scaler', preprocessing.StandardScaler()), ('clf', clf)])
## validation curve
validation_curve_plot(pipeline, X_train, y_train, clf_name = "Boosting", param_name="clf__n_estimators", param_range=np.arange(5,100,5), scoring= scoring_metric, cv=cv)
validation_curve_plot(pipeline, X_train, y_train, clf_name = "Boosting", param_name="clf__learning_rate", param_range=np.logspace(-4,1,4), scoring= scoring_metric, cv=cv)

## GridSearchCV
param_grid = {'clf__n_estimators':np.arange(2,60,4), 'clf__learning_rate': np.logspace(-3,1,6)}
gscv_tree, _ = GridSearchCV_result(pipeline, X_train, y_train, X_test, y_test, param_grid=param_grid, scoring= scoring_metric, cv=cv)

## Learning Curve
train_scores, test_scores = learning_curve_plot(gscv_tree, "Boosting", X_train, y_train, train_size_pct= np.linspace(0.2,1.0,5), scoring= scoring_metric, cv=cv)

#%% NN

simplefilter("ignore", category=ConvergenceWarning)  ## Disable the not converge warning message

print("Starts to fit NN...")

## validation curve
clf = MLPClassifier(random_state=0, max_iter=1000, activation = 'relu')
pipeline = Pipeline([('scaler', preprocessing.StandardScaler()), ('clf', clf)])
validation_curve_plot(pipeline, X_train, y_train, clf_name = 'NN', param_name="clf__hidden_layer_sizes", param_range= [20,40,60,80,100], scoring= scoring_metric, cv=cv)

clf = MLPClassifier(hidden_layer_sizes= (50,25,10),  random_state=0, max_iter=1000, activation = 'relu', early_stopping = False)
pipeline = Pipeline([('scaler', preprocessing.StandardScaler()), ('clf', clf)])
validation_curve_plot(pipeline, X_train, y_train, clf_name = 'NN', param_name="clf__learning_rate_init", param_range=np.logspace(-5,1,10), scoring= scoring_metric, cv=cv)
validation_curve_plot(pipeline, X_train, y_train, clf_name = 'NN', param_name="clf__alpha", param_range=np.logspace(-5,2,10), scoring= scoring_metric, cv=cv)

## GridSearchCV

param_grid = {'clf__learning_rate_init':[1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1], 'clf__alpha': np.logspace(-3,3,8)}
gscv_tree, gscv_score = GridSearchCV_result(pipeline, X_train, y_train, X_test, y_test, param_grid=param_grid, scoring= scoring_metric, cv=cv)

## Learning Curve
train_scores, test_scores = learning_curve_plot(gscv_tree, "NN", X_train, y_train, train_size_pct= np.linspace(0.2,1.0,5), scoring= scoring_metric, cv=cv)
loss_curve_plot(gscv_tree.best_estimator_['clf'])