import numpy as np
import pandas as pd

X=pd.read_csv("./ML_summer/t/value.csv",header=None)
y=pd.read_csv("./ML_summer/t/target.csv",header=None)


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
ex)
## 5. algorithm training
from sklearn.tree import DecisionTreeClassifier
dt = DecisionTreeClassifier(criterion = 'gini',
                            max_depth = 6,
                            random_state=42)

## learning curve
from sklearn.model_selection import learning_curve
train_sizes, train_scores, val_scores = \
    learning_curve(estimator = dt,
                   X = X_train_std,
                   y = y_train,
                   train_sizes = np.linspace(0.1,1,10),
                   cv=5,
                   n_jobs = -1)
    

train_mean = np.mean(train_scores, axis=1)
train_std = np.std(train_scores, axis=1)
# this val_scores are obtained from validation data in CV
val_mean = np.mean(val_scores, axis=1)
val_std = np.std(val_scores, axis=1)


"""
import matplotlib.pyplot as plt
plt.plot(train_sizes, train_mean, 
         color='blue', marker = 'o', markersize=5, label='Training accuracy')

plt.fill_between(train_sizes, 
                 train_mean + train_std, train_mean - train_std,
                 alpha = 0.15, color = 'blue')

plt.plot(train_sizes, val_mean, 
         color='green', linestyle = '--', 
         marker = 's', markersize=5, label='Validation accuracy')

plt.fill_between(train_sizes, 
                 val_mean + val_std, val_mean - val_std,
                 alpha = 0.15, color = 'green')

plt.grid()
plt.xlabel('# of training examples')
plt.ylabel('accuracy')
plt.legend(loc='lower right')
plt.ylim([0.8, 1.03])
plt.tight_layout()
plt.show()