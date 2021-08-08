import pandas as pd

X=pd.read_csv("./ML_summer/h_t/value.csv",header=None)
y=pd.read_csv("./ML_summer/h_t/target.csv",header=None)


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

"""
## 5. algorithm training
#원하는 알고리즘 선택 및 학습
#ex)
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import StratifiedKFold   

kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
dt = DecisionTreeClassifier(random_state=42)
        
depth_range = [1,2,3,4,5,6,7,8]
param_grid = [{'max_depth': depth_range, 'criterion': ['gini']}]

grid_cv = GridSearchCV(estimator = dt,
                       param_grid = param_grid,
                       scoring='accuracy',
                       cv = kfold,
                       refit = True,
                       n_jobs = -1)

#Estimator : Estimator 객체.
#이는 스키트 학습 추정기 인터페이스를 구현하는 것으로 가정합니다.
#param_grid : 사전 목록 또는 딕터
#매개 변수 이름('str')을 키 및 목록으로 사용하는 사전
#값 또는 해당 목록으로 시도할 매개 변수 설정
#스코어링: str, 호출 가능, 목록, 튜플 또는 딕트, 기본값=없음
#교차 검증된 모델의 성능을 평가하기 위한 전략
#테스트 세트.
"""

grid_cv.fit(X_train, y_train)

print('best validation score: %.3f' %grid_cv.best_score_)
print(grid_cv.best_params_)


bestModel = grid_cv.best_estimator_
# bestModel(X_train, y_train)

## 6. performance measure
from sklearn.metrics import accuracy_score, f1_score, recall_score, precision_score, roc_auc_score, confusion_matrix

y_test_pred = bestModel.predict(X_test)
y_test_prob = bestModel.predict_proba(X_test)
print("test data accuracy: %.3f" %accuracy_score(y_test, y_test_pred))
print("test data recall: %.3f" %recall_score(y_test, y_test_pred, average='macro'))
print("test data precison: %.3f" %precision_score(y_test, y_test_pred, average='macro'))
print("test data f1 score: %.3f" %f1_score(y_test, y_test_pred, average='macro'))
print("test data AUC: %.3f" %roc_auc_score(y_test, y_test_prob, multi_class='ovr'))
print("test data Confusion matrix:")
print(confusion_matrix(y_test, y_test_pred))
