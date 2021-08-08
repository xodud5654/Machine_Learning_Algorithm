import numpy as np
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
from sklearn.tree import DecisionTreeClassifier

dt = DecisionTreeClassifier(random_state=42)

from sklearn.model_selection import validation_curve
param_range = [1,2,3,4,5,6,7,8]

train_scores, val_scores = validation_curve(
                                estimator = dt,
                                X = X_train_std,
                                y = y_train,
                                param_name = 'max_depth',
                                param_range = param_range,
                                cv=5,
                                n_jobs = -1)
    

train_mean = np.mean(train_scores, axis=1)
train_std = np.std(train_scores, axis=1)
# this val_scores are obtained from validation data in CV
val_mean = np.mean(val_scores, axis=1)
val_std = np.std(val_scores, axis=1)
"""


import matplotlib.pyplot as plt
plt.plot(param_range, train_mean, 
         color='blue', marker = 'o', markersize=5, label='Training accuracy')

plt.fill_between(param_range, 
                 train_mean + train_std, train_mean - train_std,
                 alpha = 0.15, color = 'blue')

plt.plot(param_range, val_mean, 
         color='green', linestyle = '--', 
         marker = 's', markersize=5, label='Validation accuracy')

plt.fill_between(param_range, 
                 val_mean + val_std, val_mean - val_std,
                 alpha = 0.15, color = 'green')

plt.grid()
plt.xlabel('Parameter depth')
plt.ylabel('accuracy')
plt.legend(loc='lower right')
plt.ylim([0.5, 1.0])
plt.tight_layout()
plt.show()