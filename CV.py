import numpy as np
import pandas as pd

X=pd.read_csv("./ML_summer/h_t/value.csv",header=None)
y=pd.read_csv("./ML_summer/h_t/target.csv",header=None)
"""
X=pd.read_csv(,header=None)
y=pd.read_csv(,header=None)
"""
## 3. train / test data split
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test = \
            train_test_split(X,y,random_state=42,test_size=0.2,stratify=y)
            

## 4. scaling
from sklearn.preprocessing import StandardScaler
ss = StandardScaler()
ss.fit(X_train)
X_train_std = ss.transform(X_train)


## 5. algorithm training
"""
#원하는 알고리즘 선택 및 학습
#ex)
from sklearn.tree import DecisionTreeClassifier
dt = DecisionTreeClassifier(criterion = 'gini',
                            max_depth = 6,
                            random_state=42)

from sklearn.model_selection import cross_val_score 
scores = cross_val_score(estimator = dt,
                         X = X_train_std,
                         y = y_train,
                         cv = 5,
                         n_jobs = -1)
#estimator : 위에서 만든 모델, cv : 겹, n_jobs : 코어 사용량
"""
print("CV accuracy: %s" %scores)
print('\n')
print('CV mean accuracy: %.3f +/- %.3f' %(np.mean(scores), np.std(scores)))