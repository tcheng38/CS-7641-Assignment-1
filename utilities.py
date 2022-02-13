# -*- coding: utf-8 -*-
"""
Created on Sat Jan 22 08:51:21 2022

@author: cheng164
"""

import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt
import seaborn as sns
import time

from sklearn import preprocessing
from sklearn import svm
from sklearn import tree

from sklearn.metrics import accuracy_score, roc_auc_score, classification_report, plot_roc_curve, balanced_accuracy_score
from sklearn.metrics import f1_score
from sklearn.metrics import confusion_matrix
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.neighbors import KNeighborsClassifier

from sklearn.model_selection import validation_curve
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import learning_curve
from sklearn.model_selection import KFold
from sklearn.model_selection import train_test_split
from yellowbrick.classifier.rocauc import roc_auc


## Citation: https://scikit-learn.org/stable/auto_examples/tree/plot_cost_complexity_pruning.html#sphx-glr-auto-examples-tree-plot-cost-complexity-pruning-py
def tree_accuracy_vs_alpha(X_train, y_train, X_test, y_test):
    clf = tree.DecisionTreeClassifier(random_state=0)
    path = clf.cost_complexity_pruning_path(X_train, y_train)
    ccp_alphas, impurities = path.ccp_alphas, path.impurities
    
    clfs = []
    for ccp_alpha in ccp_alphas:
        clf = tree.DecisionTreeClassifier(random_state=0, ccp_alpha=ccp_alpha)
        clf.fit(X_train, y_train)
        clfs.append(clf)
    
    print(
        "Number of nodes in the last tree is: {} with ccp_alpha: {}".format(
            clfs[-1].tree_.node_count, ccp_alphas[-1]
        )
    )
    
     # Remove the last element in clfs and ccp_alphas, because it is the trivial tree with only one node. 
    clfs = clfs[:-1]
    ccp_alphas = ccp_alphas[:-1]
    
    train_scores = [clf.score(X_train, y_train) for clf in clfs]
    test_scores = [clf.score(X_test, y_test) for clf in clfs]
    
    fig, ax = plt.subplots()
    ax.set_xlabel("alpha")
    ax.set_ylabel("accuracy")
    ax.set_title("Accuracy vs alpha for training and testing sets")
    ax.plot(ccp_alphas, train_scores, marker="o", label="train", drawstyle="steps-post")
    ax.plot(ccp_alphas, test_scores, marker="o", label="test", drawstyle="steps-post")
    ax.legend()
    plt.show()
    

def validation_curve_plot(model, X_train, y_train, clf_name, param_name, param_range, scoring, cv):
    
    train_scores, test_scores = validation_curve(model, X_train, y_train, param_range=param_range, param_name=param_name, scoring =scoring, cv=cv)
    
    plt.figure()
    
    if clf_name == "SVM" or clf_name == "NN" or param_name =='clf__learning_rate':
        plt.semilogx(param_range, np.mean(train_scores, axis=1), label='Train Score')
        plt.semilogx(param_range, np.mean(test_scores, axis=1), label='CV Score') 
    else:       
        plt.plot(param_range, np.mean(train_scores, axis=1), label='Train Score')
        plt.plot(param_range, np.mean(test_scores, axis=1), label='CV Score')
    
    plt.legend(loc="best")
    plt.title("Validation Curve with {} ({})".format(param_name, clf_name))
    plt.xlabel(param_name)
    plt.ylabel("Score")
    plt.grid()
    plt.show()
    #plt.savefig("results/{}_model_complexity_{}_{}.png".format(clf_name, dataset_name, param_name))
    
    

def GridSearchCV_result(model, X_train, y_train, X_test, y_test, param_grid, scoring, cv , classes=None):

    gscv_clf = GridSearchCV(estimator = model, param_grid=param_grid, scoring=scoring, cv=cv)
    
    start_time_train = time.time()
    gscv_clf.fit(X_train, y_train)
    end_time_train = time.time()
    
    start_time_test = time.time()
    y_pred = gscv_clf.predict(X_test)
    end_time_test = time.time()
    print('Model Training time(s):', end_time_train - start_time_train, 'Model Prediction time(s):', end_time_test - start_time_test)
    
    print("Best Hyperparameters are: {} ; Best Mean CV Score: {}".format(gscv_clf.best_params_ , gscv_clf.best_score_))
    
    print("Confusion Matrix: \n", confusion_matrix(y_test, y_pred))
    
    cnf_matrix = confusion_matrix(y_test, y_pred)
    plt.figure()
    p = sns.heatmap(pd.DataFrame(cnf_matrix), annot=True, cmap="YlGnBu" ,fmt='g')
    p.set(xlabel='Predictable label', ylabel='Actual label', title = 'Confusion Matrix')


    # ROC & AUC
    y_pred_probas = gscv_clf.predict_proba(X_test)
    
    if y_test.nunique()>2:   # multiclass case
        auc = roc_auc_score(y_test, y_pred_probas, multi_class='ovo', average='macro') 
        plt.figure()
        roc_auc(gscv_clf, X_train, y_train, X_test, y_test, classes= classes) 
    else:
        auc = roc_auc_score(y_test, y_pred_probas[:, 1])
        plot_roc_curve(gscv_clf, X_test, y_test)  # plot ROC curves 

    if scoring == 'balanced_accuracy':
        balanced_accuracy = balanced_accuracy_score(y_test, y_pred)
        print("Balanced Accuracy of the best model:", balanced_accuracy)
        
    classifier_accuracy = accuracy_score(y_test, y_pred)
    print("Accuracy of the best model:", classifier_accuracy)
    print("Model AUC:", auc)
    print('Classification Report: \n', classification_report(y_test, y_pred))
    
    return gscv_clf, gscv_clf.best_score_ 

 
 
def learning_curve_plot(model, clf_name, X_train, y_train, train_size_pct, scoring, cv):

    train_sizes, train_scores, test_scores = learning_curve(model, X_train, y_train, train_sizes=train_size_pct, scoring=scoring, cv=cv)
    
    plt.figure()
    plt.plot(train_sizes, np.mean(train_scores, axis=1), label='Train Score')
    plt.plot(train_sizes, np.mean(test_scores, axis=1), label='CV Score')
    plt.legend()
    plt.title("Learning Curve for {}".format(clf_name))
    plt.xlabel("Training Examples")
    plt.ylabel("Score")
    plt.grid()
    plt.show()
    
    
    return train_scores, test_scores
    

def loss_curve_plot(mlp_clf):
    plt.figure()
    plt.plot(mlp_clf.loss_curve_)
    plt.title("Loss Curve", fontsize=14)
    plt.xlabel('Iterations')
    plt.ylabel('Cost')
    plt.show()