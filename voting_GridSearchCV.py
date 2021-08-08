import pandas as pd

X=pd.read_csv("./t/value.csv",header=None)
y=pd.read_csv("./t/target.csv",header=None)

"""
## 2. data preprocessing: abnormal data check (NaN, outlier check!!!)

from sklearn.preprocessing import LabelEncoder

le = LabelEncoder()
y = le.fit_transform(y)

"""
## 3. train / validation / test data split
from sklearn.model_selection import train_test_split

X_train,X_test,y_train,y_test = \
            train_test_split(X,y,random_state=42,test_size=0.2,stratify=y)
            


## 4. data preprocessing
from sklearn.preprocessing import StandardScaler
ss = StandardScaler()
ss.fit(X_train)
X_train_std = ss.transform(X_train)
X_test_std = ss.transform(X_test)


## 5. algorithm training
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import StratifiedKFold   

kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

## model 1(linear)
clf1 = SVC(kernel='linear', random_state=42, probability=True)

C_range = [0.0001, 0.001, 0.01, 0.1, 1.0, 10.0, 100.0, 1000.0]
svm_param_grid = [{'C': C_range}]

svm_grid_cv = GridSearchCV(estimator = clf1,
                       param_grid = svm_param_grid,
                       scoring='accuracy',
                       cv = kfold,
                       refit = True,
                       n_jobs=-1)

svm_grid_cv.fit(X_train_std, y_train)

print('best validation score: %.3f' %svm_grid_cv.best_score_)
print(svm_grid_cv.best_params_)

svm_bestModel = svm_grid_cv.best_estimator_


## model 2(logistic)
clf2 = LogisticRegression(random_state=42)

C_range = [0.0001, 0.001, 0.01, 0.1, 1.0, 10.0, 100.0, 1000.0]
lr_param_grid = [{'C': C_range}]
lr_grid_cv = GridSearchCV(estimator = clf2,
                       param_grid = lr_param_grid,
                       scoring='accuracy',
                       cv = kfold,
                       refit = True,
                       n_jobs=-1)

lr_grid_cv.fit(X_train_std, y_train)

print('best validation score: %.3f' %lr_grid_cv.best_score_)
print(lr_grid_cv.best_params_)

lr_bestModel = lr_grid_cv.best_estimator_


## model 3(knn)
clf3 = KNeighborsClassifier(p=2, metric='minkowski')

k_range = [3,5,7,9]
k_param_grid = [{'n_neighbors': k_range}]
knn_grid_cv = GridSearchCV(estimator = clf3,
                       param_grid = k_param_grid,
                       scoring='accuracy',
                       cv = kfold,
                       refit = True,
                       n_jobs=-1)

knn_grid_cv.fit(X_train_std, y_train)

print('best validation score: %.3f' %knn_grid_cv.best_score_)
print(knn_grid_cv.best_params_)

knn_bestModel = knn_grid_cv.best_estimator_


from sklearn.ensemble import VotingClassifier

clf_voting = VotingClassifier(estimators=[
                                ('svm',svm_grid_cv),
                                ('lr', lr_grid_cv),
                                ('knn',knn_grid_cv)],
                            voting = 'soft',
                            weights = [1,1,1],
                            n_jobs=-1)
        

clf_voting.fit(X_train_std, y_train)

## 6. performance measure
from sklearn.metrics import accuracy_score, f1_score, recall_score, precision_score, roc_auc_score, confusion_matrix

y_test_pred = clf_voting.predict(X_test_std)
y_test_prob = clf_voting.predict_proba(X_test_std)
print("test data accuracy: %.3f" %accuracy_score(y_test, y_test_pred))
print("test data recall: %.3f" %recall_score(y_test, y_test_pred, average='macro'))
print("test data precison: %.3f" %precision_score(y_test, y_test_pred, average='macro'))
print("test data f1 score: %.3f" %f1_score(y_test, y_test_pred, average='macro'))
print("test data AUC: %.3f" %roc_auc_score(y_test, y_test_prob, multi_class='ovr'))
print("test data Confusion matrix:")
print(confusion_matrix(y_test, y_test_pred))



    
    
    
    









