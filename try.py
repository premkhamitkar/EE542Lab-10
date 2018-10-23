#import pandas  as pd
import matplotlib.pyplot as plt
#import numpy as np
import pandas as pd 
import hashlib
import imblearn
import os 
from utils import logger
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import numpy as np


from sklearn.feature_selection import SelectFromModel
from sklearn import datasets
from sklearn.linear_model import LassoCV
from sklearn.linear_model import Lasso
from sklearn.model_selection import KFold
from sklearn.model_selection import GridSearchCV

from utils import logger

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.svm import SVC
from sklearn.metrics import precision_score
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
from sklearn.metrics import recall_score

#############

data = pd.read_csv('C:/Users/Aswathi/Downloads/miRNA_matrix.csv')
data['label'] = data['label'].fillna(4).astype(int)
#data.head(3)

####################

pd.value_counts(data['label']).plot.bar()
plt.title('Histogram')
plt.xlabel('Class')
plt.ylabel('Frequency')
data['label'].value_counts()

#######################

from sklearn.preprocessing import StandardScaler

data = data.drop(['file_id'], axis=1)



#####################

X = np.array(data.ix[:, data.columns != 'label'])
y = np.array(data.ix[:, data.columns == 'label'])

print('Shape of X: {}'.format(X.shape))
print('Shape of y: {}'.format(y.shape))

#######################

from imblearn.over_sampling import SMOTE

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)
print("Number transactions X_train dataset: ", X_train.shape)
print("Number transactions y_train dataset: ", y_train.shape)
print("Number transactions X_test dataset: ", X_test.shape)
print("Number transactions y_test dataset: ", y_test.shape)
####


scaler = StandardScaler()
scaler.fit(X_train)
X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)

##########################
from sys import getsizeof
print("Before UnderSampling, counts of label '0': {}".format(sum(y_train==0)))
print("Before UnderSampling, counts of label '1': {}".format(sum(y_train==1)))
print("Before UnderSampling, counts of label '2': {}".format(sum(y_train==2)))
print("Before UnderSampling, counts of label '3': {}".format(sum(y_train==3)))
print("Before UnderSampling, counts of label 'NaN': {} \n".format(sum(y_train==4)))

from imblearn.under_sampling import RandomUnderSampler
sm = RandomUnderSampler()
X_train = X_train.astype(int)
X_train_res, y_train_res = sm.fit_sample(X_train, y_train)

#print('After OverSampling, the shape of train_X: {}'.format(X_train_res.shape))
#print('After OverSampling, the shape of train_y: {} \n'.format(y_train_res.shape))

print("After UnderSampling, counts of label '0': {}".format(sum(y_train_res==0)))
print("After UnderSampling, counts of label '1': {}".format(sum(y_train_res==1)))
print("After UnderSampling, counts of label '2': {}".format(sum(y_train_res==2)))
print("After UnderSampling, counts of label '3': {}".format(sum(y_train_res==3)))
print("After UnderSampling, counts of label 'NaN': {}".format(sum(y_train_res==4)))

#############################

from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, precision_recall_curve, auc, roc_auc_score, roc_curve, recall_score, classification_report

parameters = {
    'C': np.linspace(1, 10, 10)
             }
lr = LogisticRegression(solver = 'saga', multi_class='auto')
clf = GridSearchCV(lr, parameters, cv=5, verbose=5, n_jobs=3)
clf.fit(X_train_res, y_train_res)

####################

clf.best_params_

##################

lr1 = LogisticRegression(C=4,penalty='l1', verbose=5,solver = 'saga', multi_class='auto')
lr1.fit(X_train_res, y_train_res.ravel())

###############

import itertools

def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=0)
    plt.yticks(tick_marks, classes)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        #print("Normalized confusion matrix")
    else:
        1#print('Confusion matrix, without normalization')

    #print(cm)

    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, cm[i, j],
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')

####################

y_train_pre = lr1.predict(X_train)

cnf_matrix_tra = confusion_matrix(y_train, y_train_pre)

print("Recall metric in the train dataset: {}%".format(100*cnf_matrix_tra[1,1]/(cnf_matrix_tra[1,0]+cnf_matrix_tra[1,1])))


class_names = [0,1,2,3,4]
plt.figure()
plot_confusion_matrix(cnf_matrix_tra , classes=class_names, title='Train Data Confusion matrix')
plt.show()

#######################

y_pre = lr1.predict(X_test)

cnf_matrix = confusion_matrix(y_test, y_pre)

print("Recall metric in the testing dataset: {}%".format(100*cnf_matrix[1,1]/(cnf_matrix[1,0]+cnf_matrix[1,1])))
#print("Precision metric in the testing dataset: {}%".format(100*cnf_matrix[0,0]/(cnf_matrix[0,0]+cnf_matrix[1,0])))
# Plot non-normalized confusion matrix
class_names = [0,1,2,3,4]
plt.figure()
plot_confusion_matrix(cnf_matrix , classes=class_names, title='Test Data Confusion matrix')
plt.show()

####################

tmp = lr1.fit(X_train_res, y_train_res)
#####################
from sklearn.preprocessing import label_binarize
from itertools import cycle


#y_pred_sample_score = tmp.decision_function(X_test)
#print(y_pred_sample_score.shape)
y_pre = label_binarize(y_pre, classes=[0,1,2,3,4])
y_test = label_binarize(y_test, classes=[0,1,2,3,4])
n_classes= 5
# Compute ROC curve and ROC area for each class
fpr = dict()
tpr = dict()
roc_auc = dict()
for i in range(n_classes):
    fpr[i], tpr[i], _ = roc_curve(y_test[:, i], y_pre[:, i])
    roc_auc[i] = auc(fpr[i], tpr[i])

#fpr, tpr, thresholds = roc_curve(y_test, y_pred_sample_score)

lw=1 #roc_auc = auc(fpr,tpr)

colors = cycle(['aqua', 'darkorange', 'cornflowerblue', 'red', 'blue'])
for i, color in zip(range(5), colors):
    plt.plot(fpr[i], tpr[i], color=color, lw=lw,
             label='ROC curve of class {0} (area = {1:0.2f})'
             ''.format(i, roc_auc[i]))

# Plot ROC
plt.title('Receiver Operating Characteristic')
#plt.plot(fpr, tpr, 'b',label='AUC = %0.3f'% roc_auc)
plt.legend(loc='lower right')
plt.plot([0,1],[0,1],'k--', lw=lw)
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
plt.show()

##############

roc_auc

###############
