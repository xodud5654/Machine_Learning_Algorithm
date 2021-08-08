import pandas as pd

X=pd.read_csv("./t/value.csv",header=None)
y=pd.read_csv("./t/target.csv",header=None)


## 2. data preprocessing: abnormal data check (NaN, outlier check!!!)


## 3. train / validation / test data split
from sklearn.model_selection import train_test_split

X_train,X_test,y_train,y_test = \
            train_test_split(X,y,random_state=42,test_size=0.2,stratify=y)

            
## 4. scaling
from sklearn.preprocessing import StandardScaler
ss = StandardScaler()
ss.fit(X_train)
X_train_std = ss.transform(X_train)
X_test_std = ss.transform(X_test)


## 5. algorithm training
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier


    #3개의 모델을 만든다.
clf1 = SVC(kernel='linear', random_state=42, probability=True)
clf2 = LogisticRegression(random_state=42)
clf3 = KNeighborsClassifier(n_neighbors=5,p=2, metric='minkowski')


from sklearn.ensemble import VotingClassifier

clf_voting = VotingClassifier(estimators=[
                                ('svm',clf1),
                                ('lr', clf2),
                                ('knn',clf3)],
                            voting = 'hard',
                            weights = [1,1,1])
        
clf_voting.fit(X_train_std, y_train)
        


## 6. performance measure
from sklearn.metrics import accuracy_score, f1_score, recall_score, precision_score, roc_auc_score, confusion_matrix

y_test_pred = clf_voting.predict(X_test_std)            #voting = 'hard'
#y_test_prob = clf_voting.predict_proba(X_test_std)     #votiong = 'soft'

print("test data accuracy: %.3f" %accuracy_score(y_test, y_test_pred))
print("test data recall: %.3f" %recall_score(y_test, y_test_pred, average='macro'))
print("test data precison: %.3f" %precision_score(y_test, y_test_pred, average='macro'))
print("test data f1 score: %.3f" %f1_score(y_test, y_test_pred, average='macro'))       #voting = 'hard'
#print("test data AUC: %.3f" %roc_auc_score(y_test, y_test_prob, multi_class='ovr'))    #votiong = 'soft'
print("test data Confusion matrix:")
print(confusion_matrix(y_test, y_test_pred))

