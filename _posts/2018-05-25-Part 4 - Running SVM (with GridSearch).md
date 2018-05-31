
# `Part 4: Building Models (cont'd)`


```python
import pandas as pd
import numpy as np
```


```python
bank_dataset = pd.read_csv('bank dataset (cleaned).csv')
```


```python
bank_dataset = bank_dataset.drop(labels=['Unnamed: 0','days_passed'],axis=1)
```

## Preparing our Predictor & Target variables


```python
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from imblearn.combine import SMOTEENN
```


```python
y = bank_dataset['subscription']
X = bank_dataset.drop(labels=['subscription','age','employees','e3m'],axis=1)
# X columns dropped based on feature selection.
```


```python
# Dummy encode categories with more than 2 outcomes.
X_dummed = pd.get_dummies(X, columns= ['occupation','marital','education','housing_loan','personal_loan','contact','month','day','prev_outcome'], drop_first=True)
```


```python
# Standard Scaler.
ss = StandardScaler()
X_scaled = ss.fit_transform(X_dummed)
```


```python
# Train-test split.
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.35, random_state=8)
```


```python
# SMOTEENN combination of over- & under- sampling.
smote_enn = SMOTEENN(random_state=8)
X_trainresam, y_trainresam = smote_enn.fit_sample(X_train, y_train)
```


```python
# Counting the y output variables.
from collections import Counter
print(sorted(Counter(y_trainresam).items()))
```

    [(0, 14002), (1, 16306)]


### New Baseline Accuracy (following balancing of dataset)


```python
np.mean(y_trainresam)
```




    0.5380097663983107



### Model 6: SVM + Gridsearch


```python
from sklearn import svm
from sklearn.model_selection import GridSearchCV
from sklearn import metrics
from sklearn.metrics import classification_report
```


```python
svc_a = svm.SVC()

gamma_range = np.logspace(-4, 3, 3)
C_range = np.logspace(-2, 3, 3)
kernel_range = ['rbf', 'sigmoid', 'linear', 'poly']

param_grid = dict(gamma=gamma_range, C=C_range, kernel=kernel_range)

grid = GridSearchCV(svc_a, param_grid, cv=5, scoring='accuracy', verbose=1)
```


```python
grid.fit(X_trainresam, y_trainresam)
```

    Fitting 5 folds for each of 36 candidates, totalling 180 fits



```python
# Cross-validated best accuracy score.
print 'best parameters'
print grid.best_params_
print 'best score achieved'
print grid.best_score_
```


```python
best_grid = grid.best_estimator_
```


```python
# Deriving accuracy score on test set.
y_pred = best_grid.predict(X_test)
print ('accuracy = {}'.format(metrics.accuracy_score(y_test, y_pred)))
```


```python
# Deriving accuracy score alternative?
best_grid.score(X_test, y_test)
```


```python
# Derive classification report.
print classification_report(y_test, y_pred)
```


```python
# Compute ROC_AUC score.
metrics.roc_auc_score(y_test, y_pred)
```
