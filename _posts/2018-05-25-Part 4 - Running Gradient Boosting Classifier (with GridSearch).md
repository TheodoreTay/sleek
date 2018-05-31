
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



### Model 8: Gradient Boosting


```python
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import cross_val_score
from sklearn import metrics
from sklearn.metrics import classification_report
```


```python
GBC = GradientBoostingClassifier()
GBC.fit(X_trainresam, y_trainresam)
```




    GradientBoostingClassifier(criterion='friedman_mse', init=None,
                  learning_rate=0.1, loss='deviance', max_depth=3,
                  max_features=None, max_leaf_nodes=None,
                  min_impurity_decrease=0.0, min_impurity_split=None,
                  min_samples_leaf=1, min_samples_split=2,
                  min_weight_fraction_leaf=0.0, n_estimators=100,
                  presort='auto', random_state=None, subsample=1.0, verbose=0,
                  warm_start=False)




```python
# Cross-validated best accuracy score.
accuracy_scores = cross_val_score(GBC, X_trainresam, y_trainresam, cv=10, scoring='accuracy')

print 'accuracy_scores'
print accuracy_scores
print '--------'
print np.mean(accuracy_scores)
```

    accuracy_scores
    [0.9224934  0.95316623 0.96898713 0.96634774 0.9643682  0.96601782
     0.9679868  0.96831683 0.97062706 0.96468647]
    --------
    0.9612997676519666



```python
# Deriving accuracy score on test set.
y_pred = GBC.predict(X_test)
print ('accuracy = {}'.format(metrics.accuracy_score(y_test, y_pred)))
```

    accuracy = 0.865366585412



```python
# Deriving accuracy score alternative?
GBC.score(X_test, y_test)
```




    0.8653665854115882




```python
# Derive classification report.
print classification_report(y_test, y_pred)
```

                 precision    recall  f1-score   support
    
              0       0.98      0.86      0.92      9271
              1       0.49      0.88      0.63      1395
    
    avg / total       0.92      0.87      0.88     10666
    



```python
# Compute ROC_AUC score.
metrics.roc_auc_score(y_test, y_pred)
```




    0.8710951674566972



__Note:__
<br> As this model is selected to be the best out of all, we will be constructing a confusion matrix, analysing to see what is the value of the false negatives, and make an attempt to reduce it,


```python
from sklearn.metrics import confusion_matrix
```


```python
# Constructing a confusion matrix.
```


```python
conmat = np.array(confusion_matrix(y_test, y_pred, labels=[0,1]))

confusion_table = pd.DataFrame(conmat, index=['did_not', 'subscribed'],
                         columns=['predicted_did_not','predicted_subscribed'])
confusion_table
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>predicted_did_not</th>
      <th>predicted_subscribed</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>did_not</th>
      <td>8004</td>
      <td>1267</td>
    </tr>
    <tr>
      <th>subscribed</th>
      <td>169</td>
      <td>1226</td>
    </tr>
  </tbody>
</table>
</div>




```python
# Lowering the threshold.
```


```python
y_pp = pd.DataFrame(GBC.predict_proba(X_test), columns=['class_0_pp','class_1_pp'])
y_pp.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>class_0_pp</th>
      <th>class_1_pp</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0.992995</td>
      <td>0.007005</td>
    </tr>
    <tr>
      <th>1</th>
      <td>0.981713</td>
      <td>0.018287</td>
    </tr>
    <tr>
      <th>2</th>
      <td>0.792149</td>
      <td>0.207851</td>
    </tr>
    <tr>
      <th>3</th>
      <td>0.995374</td>
      <td>0.004626</td>
    </tr>
    <tr>
      <th>4</th>
      <td>0.993962</td>
      <td>0.006038</td>
    </tr>
  </tbody>
</table>
</div>




```python
cutoffs = [0.1, 0.2, 0.3, 0.4, 0.5]
col_names = ['0.1','0.2','0.3','0.4', '0.5']

for index, threshold in enumerate(cutoffs):
    y_pp[col_names[index]] = [1.0 if values >= threshold else 0.0 for values in y_pp['class_1_pp']]
    
    conmat = np.array(confusion_matrix(y_test, y_pp[col_names[index]], labels=[0,1]))
    
    confusion_table = pd.DataFrame(conmat, index=['did_not', 'subscribed'],
                         columns=['predicted_did_not','predicted_subscribed'])
    
    print 'confusion table for cut-off values {}'.format(col_names[index])
    print confusion_table
    print '--------------'
```

    confusion table for cut-off values 0.1
                predicted_did_not  predicted_subscribed
    did_not                  6855                  2416
    subscribed                 37                  1358
    --------------
    confusion table for cut-off values 0.2
                predicted_did_not  predicted_subscribed
    did_not                  7334                  1937
    subscribed                 74                  1321
    --------------
    confusion table for cut-off values 0.3
                predicted_did_not  predicted_subscribed
    did_not                  7613                  1658
    subscribed                109                  1286
    --------------
    confusion table for cut-off values 0.4
                predicted_did_not  predicted_subscribed
    did_not                  7827                  1444
    subscribed                134                  1261
    --------------
    confusion table for cut-off values 0.5
                predicted_did_not  predicted_subscribed
    did_not                  8004                  1267
    subscribed                169                  1226
    --------------



```python
y_pp.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>class_0_pp</th>
      <th>class_1_pp</th>
      <th>0.1</th>
      <th>0.2</th>
      <th>0.3</th>
      <th>0.4</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0.992995</td>
      <td>0.007005</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>0.981713</td>
      <td>0.018287</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>0.792149</td>
      <td>0.207851</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>0.995374</td>
      <td>0.004626</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>0.993962</td>
      <td>0.006038</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
  </tbody>
</table>
</div>



### Model 9: Gradient Boosting + Gridsearch

To start with, I’ll test max_depth values of 5 to 15 in steps of 2 and min_samples_split from 200 to 1000 in steps of 200. These are just based on my intuition. You can set wider ranges as well and then perform multiple iterations for smaller ranges.


```python
param = {'max_depth':[None,3,6,9], 'min_samples_split':range(100,901,200)}

GBC_gs = GridSearchCV(estimator = GradientBoostingClassifier(learning_rate=0.1, n_estimators=15, max_features='sqrt', subsample=0.8, random_state=8), 
param_grid = param, scoring='accuracy',n_jobs=4,iid=False, cv=5)
```


```python
GBC_gs.fit(X_trainresam, y_trainresam)
```




    GridSearchCV(cv=5, error_score='raise',
           estimator=GradientBoostingClassifier(criterion='friedman_mse', init=None,
                  learning_rate=0.1, loss='deviance', max_depth=3,
                  max_features='sqrt', max_leaf_nodes=None,
                  min_impurity_decrease=0.0, min_impurity_split=None,
                  min_samples_leaf=1, min_samples_split=2,
                  min_weight_fraction_leaf=0.0, n_estimators=15,
                  presort='auto', random_state=8, subsample=0.8, verbose=0,
                  warm_start=False),
           fit_params=None, iid=False, n_jobs=4,
           param_grid={'min_samples_split': [100, 300, 500, 700, 900], 'max_depth': [None, 3, 6, 9]},
           pre_dispatch='2*n_jobs', refit=True, return_train_score='warn',
           scoring='accuracy', verbose=0)




```python
# Cross-validated best accuracy score.
print 'best parameters'
print GBC_gs.best_params_
print 'best score achieved'
print GBC_gs.best_score_
```

    best parameters
    {'min_samples_split': 100, 'max_depth': None}
    best score achieved
    0.9652574820733468



```python
best_GBC = GBC_gs.best_estimator_
```


```python
# Deriving accuracy score on test set.
y_pred = best_GBC.predict(X_test)
print ('accuracy = {}'.format(metrics.accuracy_score(y_test, y_pred)))
```

    accuracy = 0.860866304144



```python
# Deriving accuracy score alternative?
best_GBC.score(X_test, y_test)
```




    0.860866304144009




```python
# Derive classification report.
print classification_report(y_test, y_pred)
```

                 precision    recall  f1-score   support
    
              0       0.98      0.86      0.91      9271
              1       0.48      0.88      0.62      1395
    
    avg / total       0.91      0.86      0.88     10666
    



```python
# Compute ROC_AUC score.
metrics.roc_auc_score(y_test, y_pred)
```




    0.8703333978966283


