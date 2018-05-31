
# `Part 4: Building Models`


```python
import pandas as pd
import numpy as np
```


```python
bank_dataset = pd.read_csv('bank dataset (cleaned).csv')
```


```python
bank_dataset.columns
```




    Index([u'Unnamed: 0', u'age', u'occupation', u'marital', u'education',
           u'housing_loan', u'personal_loan', u'contact', u'month', u'day',
           u'duration', u'contact_freq', u'days_passed', u'contact_bef',
           u'prev_outcome', u'emp_var_rate', u'cpi_index', u'cci_index', u'e3m',
           u'employees', u'subscription', u'prev_part'],
          dtype='object')




```python
bank_dataset = bank_dataset.drop(labels=['Unnamed: 0','days_passed'],axis=1)
```


```python
# Create a continuous variable dataframe.
# We would like to conduct a Pearson's correlation to identify for any potential correlation prior to modelling.
# This is a very basic & raw feature selection step.
continuous = {}
for cols in bank_dataset.columns:
        if bank_dataset[cols].dtypes == int:
            continuous[cols] = bank_dataset[cols]
        elif bank_dataset[cols].dtypes == 'float64':
            continuous[cols] = bank_dataset[cols]
        else:
            pass
```


```python
continuous = pd.DataFrame(continuous)
```


```python
#Drop numerical categorical columns, except 'subscription'.
continuous = continuous.drop(labels=['prev_part'],axis=1)
```


```python
continuous['subscription'] = bank_dataset['subscription']
```


```python
# correlation coefficients.
pearsons_table = continuous.corr(method='pearson')
```


```python
pearsons_table[(pearsons_table>0.5) | (pearsons_table<-0.5)]
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
      <th>age</th>
      <th>cci_index</th>
      <th>contact_bef</th>
      <th>contact_freq</th>
      <th>cpi_index</th>
      <th>duration</th>
      <th>e3m</th>
      <th>emp_var_rate</th>
      <th>employees</th>
      <th>subscription</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>age</th>
      <td>1.0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>cci_index</th>
      <td>NaN</td>
      <td>1.0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>contact_bef</th>
      <td>NaN</td>
      <td>NaN</td>
      <td>1.0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>contact_freq</th>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>1.0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>cpi_index</th>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>1.000000</td>
      <td>NaN</td>
      <td>0.667198</td>
      <td>0.765986</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>duration</th>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>1.0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>e3m</th>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>0.667198</td>
      <td>NaN</td>
      <td>1.000000</td>
      <td>0.969408</td>
      <td>0.944864</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>emp_var_rate</th>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>0.765986</td>
      <td>NaN</td>
      <td>0.969408</td>
      <td>1.000000</td>
      <td>0.900361</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>employees</th>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>0.944864</td>
      <td>0.900361</td>
      <td>1.000000</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>subscription</th>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>1.0</td>
    </tr>
  </tbody>
</table>
</div>



__Comments:__
- As we can see, there is a strong positive correlation between (e3m & emp_var_rate/employees) AND (emp_var_rate & employees).
- This is largely due to these factors having a strong influence on one another by definition.
- We will not drop them just yet (until we have conducted further feature selection).

***
***
***

## Step 1: Preliminary modelling (without up or down sampling)


```python
y = bank_dataset['subscription']
X = bank_dataset.drop(labels='subscription',axis=1)
```


```python
y.value_counts()
```




    0    26616
    1     3858
    Name: subscription, dtype: int64



### Baseline model accuracy


```python
# This is what you will be comparing your model accuracy against.
1.0 - np.mean(y)
```




    0.8734002756448119



### Dummy encode, Standard Scale & Train-Test Split


```python
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
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

### Model 1: Logistic regression


```python
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from sklearn import metrics
from sklearn.metrics import classification_report
```


```python
log_reg = LogisticRegression()
```


```python
accuracy_scores = cross_val_score(log_reg, X_train, y_train, cv=10)
print 'accuracy_scores'
print accuracy_scores
print '--------'
print np.mean(accuracy_scores)
```

    accuracy_scores
    [0.89909183 0.89808274 0.89707366 0.90257446 0.89298334 0.90808081
     0.8979798  0.90353535 0.91161616 0.8969697 ]
    --------
    0.9007987851380148



```python
# Deriving accuracy score on test set.
log_reg.fit(X_train, y_train)
```




    LogisticRegression(C=1.0, class_weight=None, dual=False, fit_intercept=True,
              intercept_scaling=1, max_iter=100, multi_class='ovr', n_jobs=1,
              penalty='l2', random_state=None, solver='liblinear', tol=0.0001,
              verbose=0, warm_start=False)




```python
y_pred = log_reg.predict(X_test)
print ('accuracy = {}'.format(metrics.accuracy_score(y_test, y_pred)))
```

    accuracy = 0.899962497656



```python
y_pred = log_reg.predict(X_test)
```


```python
print classification_report(y_test, y_pred)
```

                 precision    recall  f1-score   support
    
              0       0.92      0.97      0.94      9271
              1       0.69      0.43      0.53      1395
    
    avg / total       0.89      0.90      0.89     10666
    


__Comments:__
<br>The 'Accuracy Score' of 0.899 for logistic regression model is slightly better than baseline accuracy.


```python
# Conf matrix, ROC-AUC?
```

### Model 2: KNN

Definition:
- Simple algorithm based on distances from a stipulated number of 'K' neighbours (e.g. 3, 5, 10 etc.)


```python
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_val_score
from sklearn.metrics import confusion_matrix
```


```python
knn = KNeighborsClassifier(n_neighbors=5)
```


```python
# Comparing accuracy score to baseline.
accuracy_score = cross_val_score(knn, X_train, y_train, cv=10)
print(accuracy_score)
print np.mean(accuracy_score)
```

    [0.88597376 0.88294652 0.87891019 0.88036345 0.8859162  0.88585859
     0.87424242 0.87474747 0.88232323 0.88737374]
    0.8818655585552891



```python
# Deriving accuracy score on test set.
knn.fit(X_train, y_train)
```




    KNeighborsClassifier(algorithm='auto', leaf_size=30, metric='minkowski',
               metric_params=None, n_jobs=1, n_neighbors=5, p=2,
               weights='uniform')




```python
y_pred = knn.predict(X_test)
print ('accuracy = {}'.format(metrics.accuracy_score(y_test, y_pred)))
```

    accuracy = 0.883367710482


__Comments:__
<br>The 'Accuracy Score' of 0.88 for knn model is slightly better than baseline accuracy.
<br> Chances are we will not use knn for evaluation.

***
***
***

## Step 2: Up/Down Sampling


```python
from imblearn.combine import SMOTEENN
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

    [(0, 13981), (1, 16638)]


***
***
***

## Step 3: Feature Selection

### Feature Selection (SelectKBest)

Note: The F-test is explained variance divided by unexplained variance. High numbers will result if our explained variance (what we know) is much greater than our unexplained variance (what we don't know).


```python
cols = list(X_dummed.columns)
```


```python
from sklearn.feature_selection import SelectKBest, f_classif

skb_f = SelectKBest(f_classif, k=5)
skb_f.fit(X_trainresam, y_trainresam)


kbest = pd.DataFrame([cols, list(skb_f.scores_)], 
                     index=['feature','f_classif']).T.sort_values('f_classif', ascending=False)

kbest
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
      <th>feature</th>
      <th>f_classif</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>8</th>
      <td>employees</td>
      <td>14519.8</td>
    </tr>
    <tr>
      <th>7</th>
      <td>e3m</td>
      <td>13355.6</td>
    </tr>
    <tr>
      <th>4</th>
      <td>emp_var_rate</td>
      <td>12233.7</td>
    </tr>
    <tr>
      <th>1</th>
      <td>duration</td>
      <td>10326</td>
    </tr>
    <tr>
      <th>9</th>
      <td>prev_part</td>
      <td>3873.94</td>
    </tr>
    <tr>
      <th>45</th>
      <td>prev_outcome_success</td>
      <td>3587.35</td>
    </tr>
    <tr>
      <th>30</th>
      <td>contact_telephone</td>
      <td>2993.28</td>
    </tr>
    <tr>
      <th>44</th>
      <td>prev_outcome_nonexistent</td>
      <td>2772.19</td>
    </tr>
    <tr>
      <th>3</th>
      <td>contact_bef</td>
      <td>2747.98</td>
    </tr>
    <tr>
      <th>5</th>
      <td>cpi_index</td>
      <td>1879.09</td>
    </tr>
    <tr>
      <th>36</th>
      <td>month_may</td>
      <td>1272.78</td>
    </tr>
    <tr>
      <th>38</th>
      <td>month_oct</td>
      <td>1023.29</td>
    </tr>
    <tr>
      <th>35</th>
      <td>month_mar</td>
      <td>974.268</td>
    </tr>
    <tr>
      <th>39</th>
      <td>month_sep</td>
      <td>822.109</td>
    </tr>
    <tr>
      <th>2</th>
      <td>contact_freq</td>
      <td>769.289</td>
    </tr>
    <tr>
      <th>14</th>
      <td>occupation_retired</td>
      <td>650.194</td>
    </tr>
    <tr>
      <th>10</th>
      <td>occupation_blue-collar</td>
      <td>547.701</td>
    </tr>
    <tr>
      <th>17</th>
      <td>occupation_student</td>
      <td>422.115</td>
    </tr>
    <tr>
      <th>23</th>
      <td>education_basic.9y</td>
      <td>307.549</td>
    </tr>
    <tr>
      <th>27</th>
      <td>education_university.degree</td>
      <td>300.51</td>
    </tr>
    <tr>
      <th>32</th>
      <td>month_dec</td>
      <td>266.199</td>
    </tr>
    <tr>
      <th>21</th>
      <td>marital_single</td>
      <td>233.312</td>
    </tr>
    <tr>
      <th>33</th>
      <td>month_jul</td>
      <td>170.247</td>
    </tr>
    <tr>
      <th>0</th>
      <td>age</td>
      <td>146.764</td>
    </tr>
    <tr>
      <th>37</th>
      <td>month_nov</td>
      <td>143.887</td>
    </tr>
    <tr>
      <th>20</th>
      <td>marital_married</td>
      <td>127.145</td>
    </tr>
    <tr>
      <th>16</th>
      <td>occupation_services</td>
      <td>124.561</td>
    </tr>
    <tr>
      <th>6</th>
      <td>cci_index</td>
      <td>92.793</td>
    </tr>
    <tr>
      <th>31</th>
      <td>month_aug</td>
      <td>67.4443</td>
    </tr>
    <tr>
      <th>22</th>
      <td>education_basic.6y</td>
      <td>63.8677</td>
    </tr>
    <tr>
      <th>40</th>
      <td>day_mon</td>
      <td>57.2435</td>
    </tr>
    <tr>
      <th>29</th>
      <td>personal_loan_yes</td>
      <td>49.0092</td>
    </tr>
    <tr>
      <th>11</th>
      <td>occupation_entrepreneur</td>
      <td>39.5332</td>
    </tr>
    <tr>
      <th>18</th>
      <td>occupation_technician</td>
      <td>31.0192</td>
    </tr>
    <tr>
      <th>41</th>
      <td>day_thu</td>
      <td>25.5338</td>
    </tr>
    <tr>
      <th>19</th>
      <td>occupation_unemployed</td>
      <td>21.3483</td>
    </tr>
    <tr>
      <th>15</th>
      <td>occupation_self-employed</td>
      <td>14.5476</td>
    </tr>
    <tr>
      <th>24</th>
      <td>education_high.school</td>
      <td>10.3086</td>
    </tr>
    <tr>
      <th>28</th>
      <td>housing_loan_yes</td>
      <td>9.07698</td>
    </tr>
    <tr>
      <th>26</th>
      <td>education_professional.course</td>
      <td>2.41055</td>
    </tr>
    <tr>
      <th>43</th>
      <td>day_wed</td>
      <td>0.89431</td>
    </tr>
    <tr>
      <th>25</th>
      <td>education_illiterate</td>
      <td>0.293242</td>
    </tr>
    <tr>
      <th>34</th>
      <td>month_jun</td>
      <td>0.18956</td>
    </tr>
    <tr>
      <th>13</th>
      <td>occupation_management</td>
      <td>0.0558223</td>
    </tr>
    <tr>
      <th>42</th>
      <td>day_tue</td>
      <td>0.0259538</td>
    </tr>
    <tr>
      <th>12</th>
      <td>occupation_housemaid</td>
      <td>0.00492398</td>
    </tr>
  </tbody>
</table>
</div>



### Feature Selection (RFE)


```python
from sklearn.feature_selection import RFECV
from sklearn.linear_model import LogisticRegression

selector = RFECV(log_reg, step=1, cv=10)
selector = selector.fit(X_trainresam, y_trainresam)

print selector.support_
print selector.ranking_
```

    [False  True  True  True  True  True  True  True  True  True  True False
      True False  True  True  True  True False False  True  True  True False
     False  True  True  True False  True  True  True  True  True  True  True
      True  True  True  True  True  True False False False False]
    [ 9  1  1  1  1  1  1  1  1  1  1  6  1 10  1  1  1  1  5  3  1  1  1  8
     13  1  1  1 12  1  1  1  1  1  1  1  1  1  1  1  1  1  2  7 11  4]



```python
rfecv_columns = np.array(cols)[selector.support_]
rfecv_columns
```




    array(['duration', 'contact_freq', 'contact_bef', 'emp_var_rate',
           'cpi_index', 'cci_index', 'e3m', 'employees', 'prev_part',
           'occupation_blue-collar', 'occupation_housemaid',
           'occupation_retired', 'occupation_self-employed',
           'occupation_services', 'occupation_student', 'marital_married',
           'marital_single', 'education_basic.6y', 'education_illiterate',
           'education_professional.course', 'education_university.degree',
           'personal_loan_yes', 'contact_telephone', 'month_aug', 'month_dec',
           'month_jul', 'month_jun', 'month_mar', 'month_may', 'month_nov',
           'month_oct', 'month_sep', 'day_mon', 'day_thu'], dtype='|S29')




```python
RFE_excluded = []
for columns in cols:
    if columns not in rfecv_columns:
        RFE_excluded.append(columns)
```


```python
# Excluded features.
RFE_excluded
```




    ['age',
     'occupation_entrepreneur',
     'occupation_management',
     'occupation_technician',
     'occupation_unemployed',
     'education_basic.9y',
     'education_high.school',
     'housing_loan_yes',
     'day_tue',
     'day_wed',
     'prev_outcome_nonexistent',
     'prev_outcome_success']



### Feature Selection (Lasso Penalty)


```python
from sklearn.linear_model import LogisticRegressionCV

log_rcv = LogisticRegressionCV(penalty='l1', Cs=100, cv=10, solver='liblinear')
log_rcv.fit(X_trainresam, y_trainresam)
```




    LogisticRegressionCV(Cs=100, class_weight=None, cv=10, dual=False,
               fit_intercept=True, intercept_scaling=1.0, max_iter=100,
               multi_class='ovr', n_jobs=1, penalty='l1', random_state=None,
               refit=True, scoring=None, solver='liblinear', tol=0.0001,
               verbose=0)




```python
coeffs = pd.DataFrame(log_rcv.coef_, columns=X_dummed.columns)
coeffs_t = coeffs.transpose()
coeffs_t.columns = ['lasso_coefs']
coeffs_abs = coeffs_t.abs().sort_values('lasso_coefs', ascending=False)
coeffs_abs
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
      <th>lasso_coefs</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>duration</th>
      <td>2.842104</td>
    </tr>
    <tr>
      <th>employees</th>
      <td>1.306021</td>
    </tr>
    <tr>
      <th>emp_var_rate</th>
      <td>0.957872</td>
    </tr>
    <tr>
      <th>month_may</th>
      <td>0.931206</td>
    </tr>
    <tr>
      <th>prev_part</th>
      <td>0.802033</td>
    </tr>
    <tr>
      <th>month_oct</th>
      <td>0.522649</td>
    </tr>
    <tr>
      <th>month_mar</th>
      <td>0.382632</td>
    </tr>
    <tr>
      <th>month_nov</th>
      <td>0.337371</td>
    </tr>
    <tr>
      <th>education_university.degree</th>
      <td>0.262446</td>
    </tr>
    <tr>
      <th>contact_bef</th>
      <td>0.232687</td>
    </tr>
    <tr>
      <th>occupation_retired</th>
      <td>0.197559</td>
    </tr>
    <tr>
      <th>cci_index</th>
      <td>0.135519</td>
    </tr>
    <tr>
      <th>contact_telephone</th>
      <td>0.124497</td>
    </tr>
    <tr>
      <th>education_basic.6y</th>
      <td>0.120823</td>
    </tr>
    <tr>
      <th>occupation_blue-collar</th>
      <td>0.115086</td>
    </tr>
    <tr>
      <th>personal_loan_yes</th>
      <td>0.111122</td>
    </tr>
    <tr>
      <th>cpi_index</th>
      <td>0.100778</td>
    </tr>
    <tr>
      <th>marital_single</th>
      <td>0.098369</td>
    </tr>
    <tr>
      <th>prev_outcome_nonexistent</th>
      <td>0.097807</td>
    </tr>
    <tr>
      <th>day_thu</th>
      <td>0.091565</td>
    </tr>
    <tr>
      <th>education_professional.course</th>
      <td>0.090763</td>
    </tr>
    <tr>
      <th>day_mon</th>
      <td>0.089610</td>
    </tr>
    <tr>
      <th>contact_freq</th>
      <td>0.088763</td>
    </tr>
    <tr>
      <th>occupation_student</th>
      <td>0.085133</td>
    </tr>
    <tr>
      <th>month_jun</th>
      <td>0.073357</td>
    </tr>
    <tr>
      <th>occupation_self-employed</th>
      <td>0.070101</td>
    </tr>
    <tr>
      <th>occupation_housemaid</th>
      <td>0.044940</td>
    </tr>
    <tr>
      <th>day_wed</th>
      <td>0.041400</td>
    </tr>
    <tr>
      <th>occupation_services</th>
      <td>0.040859</td>
    </tr>
    <tr>
      <th>occupation_unemployed</th>
      <td>0.030523</td>
    </tr>
    <tr>
      <th>education_illiterate</th>
      <td>0.022921</td>
    </tr>
    <tr>
      <th>marital_married</th>
      <td>0.008043</td>
    </tr>
    <tr>
      <th>occupation_entrepreneur</th>
      <td>0.006176</td>
    </tr>
    <tr>
      <th>month_jul</th>
      <td>0.005019</td>
    </tr>
    <tr>
      <th>education_basic.9y</th>
      <td>0.001129</td>
    </tr>
    <tr>
      <th>month_dec</th>
      <td>0.000589</td>
    </tr>
    <tr>
      <th>month_sep</th>
      <td>0.000507</td>
    </tr>
    <tr>
      <th>day_tue</th>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>age</th>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>month_aug</th>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>housing_loan_yes</th>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>education_high.school</th>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>occupation_technician</th>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>occupation_management</th>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>e3m</th>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>prev_outcome_success</th>
      <td>0.000000</td>
    </tr>
  </tbody>
</table>
</div>



__Comments:__
<br> Feature exclusion:
- From the lasso regularisation, we can tell that the main categories 'age' & 'e3m' have been zeroed.
- 'age' & 'prev_outcome' have been excluded via RFECV.

<br> The top 5 features for f_classif:
- 'employees', 'emp_var_rate', 'e3m', 'duration', 'prev_part'

<br> The top 5 features for lasso:
- 'duration', 'employees', 'emp_var_rate', 'month_may', 'prev_part'

<br> Based on the information provided (together with the Pearson's correlation table earlier), the columns age, employees & e3m will be removed.

***
***
***

## Step 4: Re-engineering predictor variables

### Drop columns from feature selection, one-hot encode, standard scale then SMOTEENN.


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


## Step 5: Actual modelling & Comparison (after feature selection & sampling)

### New Baseline Accuracy (following balancing of dataset)


```python
np.mean(y_trainresam)
```




    0.5380097663983107



### Model 3: Logistic Regression


```python
# Cross-validated mean accuracy score.
log_reg = LogisticRegression()
accuracy_scores = cross_val_score(log_reg, X_trainresam, y_trainresam, cv=10)
print 'accuracy_scores'
print accuracy_scores
print '--------'
print np.mean(accuracy_scores)
```

    accuracy_scores
    [0.93898417 0.93040897 0.93929396 0.93665457 0.93533487 0.93731442
     0.94323432 0.93861386 0.94455446 0.94257426]
    --------
    0.9386967860032283



```python
# Deriving accuracy score on test set.
log_reg.fit(X_trainresam, y_trainresam)
```




    LogisticRegression(C=1.0, class_weight=None, dual=False, fit_intercept=True,
              intercept_scaling=1, max_iter=100, multi_class='ovr', n_jobs=1,
              penalty='l2', random_state=None, solver='liblinear', tol=0.0001,
              verbose=0, warm_start=False)




```python
y_pred = log_reg.predict(X_test)
print ('accuracy = {}'.format(metrics.accuracy_score(y_test, y_pred)))
```

    accuracy = 0.837521095068



```python
# Deriving accuracy score alternative?
log_reg.score(X_test, y_test)
```




    0.8375210950684417




```python
# Derive classification report.
print classification_report(y_test, y_pred)
```

                 precision    recall  f1-score   support
    
              0       0.99      0.83      0.90      9271
              1       0.44      0.92      0.60      1395
    
    avg / total       0.91      0.84      0.86     10666
    



```python
# Compute ROC_AUC score.
metrics.roc_auc_score(y_test, y_pred)
```




    0.8715200094022715



__Comment__:
<br> Accuracy score of 0.835 is way higher than the new baseline score 0.538.

### Model 4: Logistic Regression + Gridsearch


```python
from sklearn.model_selection import GridSearchCV
```


```python
gs_params = {
    'penalty':['l1','l2'],
    'solver':['liblinear'],
    'C':np.logspace(0, 4, 10)
}

lr_gridsearch = GridSearchCV(LogisticRegression(), gs_params, cv=5, verbose=1)
```


```python
lr_gridsearch.fit(X_trainresam, y_trainresam)
```

    Fitting 5 folds for each of 20 candidates, totalling 100 fits


    [Parallel(n_jobs=1)]: Done 100 out of 100 | elapsed:   32.2s finished





    GridSearchCV(cv=5, error_score='raise',
           estimator=LogisticRegression(C=1.0, class_weight=None, dual=False, fit_intercept=True,
              intercept_scaling=1, max_iter=100, multi_class='ovr', n_jobs=1,
              penalty='l2', random_state=None, solver='liblinear', tol=0.0001,
              verbose=0, warm_start=False),
           fit_params=None, iid=True, n_jobs=1,
           param_grid={'penalty': ['l1', 'l2'], 'C': array([1.00000e+00, 2.78256e+00, 7.74264e+00, 2.15443e+01, 5.99484e+01,
           1.66810e+02, 4.64159e+02, 1.29155e+03, 3.59381e+03, 1.00000e+04]), 'solver': ['liblinear']},
           pre_dispatch='2*n_jobs', refit=True, return_train_score='warn',
           scoring=None, verbose=1)




```python
# Cross-validated best accuracy score.
lr_gridsearch.best_score_
```




    0.9390919889138182




```python
lr_gridsearch.best_params_
```




    {'C': 1.0, 'penalty': 'l1', 'solver': 'liblinear'}




```python
best_lr = lr_gridsearch.best_estimator_
```


```python
# Deriving accuracy score on test set.
y_pred = best_lr.predict(X_test)
print ('accuracy = {}'.format(metrics.accuracy_score(y_test, y_pred)))
```

    accuracy = 0.837427339209



```python
# Deriving accuracy score alternative?
best_lr.score(X_test, y_test)
```




    0.8374273392087005




```python
# Derive classification report.
print classification_report(y_test, y_pred)
```

                 precision    recall  f1-score   support
    
              0       0.99      0.83      0.90      9271
              1       0.44      0.92      0.60      1395
    
    avg / total       0.91      0.84      0.86     10666
    



```python
# Compute ROC_AUC score.
metrics.roc_auc_score(y_test, y_pred)
```




    0.8714660777875589



__Comment__:
<br> Accuracy score of 0.834 is > than new baseline score 0.538.

### Model 5: SVM

__Note:__
1. A linearly seperable classification method with a "hinge" loss function.
2. Decision boundary is defined by the MMH.


```python
from sklearn import svm
```


```python
# Cross-validated best accuracy score.
svc = svm.SVC()
accuracy_scores = cross_val_score(svc, X_trainresam, y_trainresam, cv=10, scoring='accuracy')

print 'accuracy_scores'
print accuracy_scores
print '--------'
print np.mean(accuracy_scores)
```

    accuracy_scores
    [0.96437995 0.96108179 0.96667766 0.96403827 0.96073903 0.96370835
     0.97194719 0.96534653 0.96633663 0.96666667]
    --------
    0.9650922083565912



```python
svc.fit(X_trainresam, y_trainresam)
```




    SVC(C=1.0, cache_size=200, class_weight=None, coef0=0.0,
      decision_function_shape='ovr', degree=3, gamma='auto', kernel='rbf',
      max_iter=-1, probability=False, random_state=None, shrinking=True,
      tol=0.001, verbose=False)




```python
# Deriving accuracy score on test set.
y_pred = svc.predict(X_test)
print ('accuracy = {}'.format(metrics.accuracy_score(y_test, y_pred)))
```

    accuracy = 0.846615413463



```python
# Deriving accuracy score alternative?
svc.score(X_test, y_test)
```




    0.8466154134633415




```python
# Derive classification report.
print classification_report(y_test, y_pred)
```

                 precision    recall  f1-score   support
    
              0       0.98      0.84      0.90      9271
              1       0.46      0.89      0.60      1395
    
    avg / total       0.91      0.85      0.87     10666
    



```python
# Compute ROC_AUC score.
metrics.roc_auc_score(y_test, y_pred)
```




    0.864571723055166



__Comment__:
<br> Accuracy score of 0.846 is > than the new baseline score 0.538.

### Model 6: SVM (with Gridsearch)


```python
# For model 6: SVM + Grid Search, onwards,
# kindly refer to the other Jupyter Notebooks
# as these methods are computationally expensive and are runned individually. Thank you.
```

__Currently:__
<br> Log reg (0.835)
<br> log reg, with GS (0.834)
<br> SVM (0.846)
<br> SVM, with GS (???)
