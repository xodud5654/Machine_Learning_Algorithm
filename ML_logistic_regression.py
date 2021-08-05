import numpy as np
import pandas as pd

## 1. data load
X=pd.read_csv("./t/value.csv",header=None)
y=pd.read_csv("./t/target.csv",header=None)


## 3. train / validation / test data split
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test = \
            train_test_split(X,y,random_state=42,test_size=0.2,stratify=y)
            

X_tr,X_val,y_tr,y_val = \
train_test_split(X_train,y_train,random_state=42,test_size=0.2,stratify=y_train)

## 4. data preprocessing
from sklearn.preprocessing import StandardScaler
ss = StandardScaler()
ss.fit(X_tr)
X_tr_std = ss.transform(X_tr)
X_val_std = ss.transform(X_val)


"""
#defalut hyperparam

## 5. algorithm training
from sklearn.linear_model import LogisticRegression
lr = LogisticRegression(C=0.01, random_state=42)
lr.fit(X_tr_std, y_tr)
y_val_pred = lr.predict(X_val_std)

## 6. performance measure
from sklearn.metrics import accuracy_score
acc = accuracy_score(y_val, y_val_pred)
print(acc)


"""
## find optimal hyperparameter
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
accuracy, c_val = [], []
for c in np.arange(-5,5):
    lr2 = LogisticRegression(C=10.0**c, random_state=42)
    lr2.fit(X_tr_std, y_tr)
    y_val_pred = lr2.predict(X_val_std)
    accuracy.append(accuracy_score(y_val, y_val_pred))
    c_val.append(2.0**c)
    
##
ss2 = StandardScaler()
ss2.fit(X_train)
X_train_std = ss2.transform(X_train)
X_test_std = ss2.transform(X_test)


c_idx = accuracy.index(max(accuracy))
lr_opt = LogisticRegression(C=c_val[c_idx], random_state=42)
lr_opt.fit(X_train_std, y_train)    
y_test_pred = lr_opt.predict(X_test_std)



#importing confusion matrix
from sklearn.metrics import confusion_matrix
confusion = confusion_matrix(y_test, y_test_pred)
print("\nperformence_measure\n")
print('Confusion Matrix\n')
print(confusion)

#importing accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
print("test data accuracy: %.3f" %accuracy_score(y_test, y_test_pred))

print('Micro Precision: {:.2f}'.format(precision_score(y_test, y_test_pred, average='micro')))
print('Micro Recall: {:.2f}'.format(recall_score(y_test, y_test_pred, average='micro')))
print('Micro F1-score: {:.2f}\n'.format(f1_score(y_test, y_test_pred, average='micro')))

print('Macro Precision: {:.2f}'.format(precision_score(y_test, y_test_pred, average='macro')))
print('Macro Recall: {:.2f}'.format(recall_score(y_test, y_test_pred, average='macro')))
print('Macro F1-score: {:.2f}\n'.format(f1_score(y_test, y_test_pred, average='macro')))

print('Weighted Precision: {:.2f}'.format(precision_score(y_test, y_test_pred, average='weighted')))
print('Weighted Recall: {:.2f}'.format(recall_score(y_test, y_test_pred, average='weighted')))
print('Weighted F1-score: {:.2f}'.format(f1_score(y_test, y_test_pred, average='weighted')))


from sklearn.metrics import classification_report,roc_auc_score

print('\nClassification Report\n')
print(classification_report(y_test, y_test_pred, target_names=['Class 1', 'Class 2', 'Class 3', "Class 4"]))
y_test_prob = lr_opt.predict_proba(X_test_std)
print("test data AUC: %.3f" %roc_auc_score(y_test, y_test_prob, multi_class='ovr'))



    
    
    
    
    
    
    
    
    
    









