
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
from sklearn.externals.six import StringIO  
from IPython.display import Image  
from sklearn.tree import export_graphviz
import pydotplus
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



### Model 7: Decision Tree Classifier


```python
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import cross_val_score
from sklearn import metrics
from sklearn.metrics import classification_report
```


```python
# set 4 trees
dtc1 = DecisionTreeClassifier(max_depth=1)
dtc2 = DecisionTreeClassifier(max_depth=2)
dtc3 = DecisionTreeClassifier(max_depth=3)
dtcN = DecisionTreeClassifier(max_depth=None)
```


```python
# fit 4 trees
dtc1.fit(X_trainresam, y_trainresam)
dtc2.fit(X_trainresam, y_trainresam)
dtc3.fit(X_trainresam, y_trainresam)
dtcN.fit(X_trainresam, y_trainresam)
```




    DecisionTreeClassifier(class_weight=None, criterion='gini', max_depth=None,
                max_features=None, max_leaf_nodes=None,
                min_impurity_decrease=0.0, min_impurity_split=None,
                min_samples_leaf=1, min_samples_split=2,
                min_weight_fraction_leaf=0.0, presort=False, random_state=None,
                splitter='best')




```python
# Cross-validated mean accuracy score.
dtc1_scores = cross_val_score(dtc1, X_trainresam, y_trainresam, cv=4, scoring='accuracy')
dtc2_scores = cross_val_score(dtc2, X_trainresam, y_trainresam, cv=4, scoring='accuracy')
dtc3_scores = cross_val_score(dtc3, X_trainresam, y_trainresam, cv=4, scoring='accuracy')
dtcN_scores = cross_val_score(dtcN, X_trainresam, y_trainresam, cv=4, scoring='accuracy')

print np.mean(dtc1_scores)
print np.mean(dtc2_scores)
print np.mean(dtc3_scores)
print np.mean(dtcN_scores)
```

    0.777880624051226
    0.7877779906292405
    0.9107504933555507
    0.9535449331179597



```python
# Deriving accuracy score on test set.
y_pred = dtcN.predict(X_test)
print ('accuracy = {}'.format(metrics.accuracy_score(y_test, y_pred)))
```

    accuracy = 0.866491655728



```python
# Deriving accuracy score alternative?
dtcN.score(X_test, y_test)
```




    0.8664916557284831




```python
# Derive classification report.
print classification_report(y_test, y_pred)
```

                 precision    recall  f1-score   support
    
              0       0.96      0.88      0.92      9271
              1       0.49      0.78      0.61      1395
    
    avg / total       0.90      0.87      0.88     10666
    



```python
# Compute ROC_AUC score.
metrics.roc_auc_score(y_test, y_pred)
```




    0.8318539833426699




```python
# Graphical illustration for decision tree (without max_depth dictated).
dot_data = StringIO()  

export_graphviz(dtcN, out_file=dot_data,  
                filled=True, rounded=True,
                special_characters=True,
                feature_names=X_dummed.columns)  

graph = pydotplus.graph_from_dot_data(dot_data.getvalue())  
Image(graph.create_png())  
```




![png](output_23_0.png)



### Model 8: Decision Tree Classifier + Gridsearch


```python
# gridsearch params
dtc_params = {
    'max_depth':[None,2,3,4],
    'min_samples_split':[2,4,5,10,15]
}

# set the gridsearch
dtc_gs = GridSearchCV(DecisionTreeClassifier(), 
                      dtc_params, 
                      cv=5, 
                      verbose=1, 
                      scoring='accuracy', 
                      n_jobs=-1)
```


```python
dtc_gs.fit(X_trainresam, y_trainresam)
```

    Fitting 5 folds for each of 20 candidates, totalling 100 fits


    [Parallel(n_jobs=-1)]: Done  42 tasks      | elapsed:    5.6s
    [Parallel(n_jobs=-1)]: Done 100 out of 100 | elapsed:    9.8s finished





    GridSearchCV(cv=5, error_score='raise',
           estimator=DecisionTreeClassifier(class_weight=None, criterion='gini', max_depth=None,
                max_features=None, max_leaf_nodes=None,
                min_impurity_decrease=0.0, min_impurity_split=None,
                min_samples_leaf=1, min_samples_split=2,
                min_weight_fraction_leaf=0.0, presort=False, random_state=None,
                splitter='best'),
           fit_params=None, iid=True, n_jobs=-1,
           param_grid={'min_samples_split': [2, 4, 5, 10, 15], 'max_depth': [None, 2, 3, 4]},
           pre_dispatch='2*n_jobs', refit=True, return_train_score='warn',
           scoring='accuracy', verbose=1)




```python
# Cross-validated best accuracy score.
print 'best parameters'
print dtc_gs.best_params_
print 'best score achieved'
print dtc_gs.best_score_
```

    best parameters
    {'min_samples_split': 2, 'max_depth': None}
    best score achieved
    0.9550283753464431



```python
best_dtc = dtc_gs.best_estimator_
```


```python
# Deriving accuracy score on test set.
y_pred = best_dtc.predict(X_test)
print ('accuracy = {}'.format(metrics.accuracy_score(y_test, y_pred)))
```

    accuracy = 0.865741608851



```python
# Deriving accuracy score alternative?
best_dtc.score(X_test, y_test)
```




    0.8657416088505532




```python
# Derive classification report.
print classification_report(y_test, y_pred)
```

                 precision    recall  f1-score   support
    
              0       0.96      0.88      0.92      9271
              1       0.49      0.78      0.60      1395
    
    avg / total       0.90      0.87      0.88     10666
    



```python
# Compute ROC_AUC score.
metrics.roc_auc_score(y_test, y_pred)
```




    0.8295955824788362


