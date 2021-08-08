import pandas as pd
## 1. data load

X=pd.read_csv("./t/value.csv",header=None)
y=pd.read_csv("./t/target.csv",header=None)


## 2. data preprocessing: abnormal data check (NaN, outlier check!!!)

from sklearn.preprocessing import LabelEncoder

le = LabelEncoder()
y = le.fit_transform(y)


## 3. train / validation / test data split
from sklearn.model_selection import train_test_split

X_train,X_test,y_train,y_test = \
            train_test_split(X,y,random_state=42,test_size=0.2,stratify=y)
            


## 4. data preprocessing
# from sklearn.preprocessing import StandardScaler
# ss = StandardScaler()
# ss.fit(X_train)
# X_train_std = ss.transform(X_train)
# X_test_std = ss.transform(X_test)


## 5. algorithm training
from sklearn.ensemble import BaggingClassifier
from sklearn.tree import DecisionTreeClassifier

tree = DecisionTreeClassifier(criterion='gini', max_depth=None, random_state=42)
#max_depth:None --> 순수한 수준의 트리

bag = BaggingClassifier(base_estimator=tree,
                        n_estimators=500, 
                        max_samples=1.0, 
                        max_features=1.0, 
                        bootstrap=True, 
                        bootstrap_features=False, 
                        n_jobs=-1, 
                        random_state=42)
# base_estimator : 모델(예측기), N_estimators : 모형의 갯수, bootstrap : 데이테 중복여부, \
# max_sample : 랜덤하게 몇 %씩 들어가는 것의 한게값을 결정 -> 줄일수록 서로 다른것을 배운 에측기 모일수 있다.\
# bootstrap_features : 하나의 예측기에 들어가는 샘들에 대해서 중복사용 여부 결정
# max_features  : 하나의 예측기가 가져갈 수 있는 최대의 칼럼 갯수


tree = tree.fit(X_train, y_train)

bag = bag.fit(X_train, y_train)



## 6. performance measure
from sklearn.metrics import accuracy_score, f1_score, recall_score, precision_score, roc_auc_score, confusion_matrix

y_train_tree_pred = tree.predict(X_train)
y_test_tree_pred = tree.predict(X_test)

tree_train = accuracy_score(y_train, y_train_tree_pred)
tree_test = accuracy_score(y_test, y_test_tree_pred)
print('결정 트리의 훈련 정확도/테스트 정확도 %.3f/%.3f' % (tree_train, tree_test))


y_train_bag_pred = bag.predict(X_train)
y_test_bag_pred = bag.predict(X_test)

bag_train = accuracy_score(y_train, y_train_bag_pred) 
bag_test = accuracy_score(y_test, y_test_bag_pred) 
print('배깅의 훈련 정확도/테스트 정확도 %.3f/%.3f' % (bag_train, bag_test))

