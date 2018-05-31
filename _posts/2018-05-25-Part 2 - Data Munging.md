
# `Part 2: Data Munging`


```python
import pandas as pd
import numpy as np
```


```python
# Reading in the csv file.
bank_dataset = pd.read_csv('./bank dataset (original).csv',delimiter=';')
```

Terminology Glossary:
1. Term deposit
<p>&#9679; refers to funds placed with a financial institution for a specified amount of time at an agreed interest rate in return.</p>
<br>
2. Employment variability rate
<p>&#9679; refers to volatility in employment/human resource as a result of factors such as fluctuation in product demand, sudden shortage of manpower etc.
<br> &#9679; generally you would prefer for this to be as low as possible.</p>
<br>
3. Euribor
<p>&#9679; A daily reference rate, based on averaged interests rates in which Eurozone banks offer to lend unsecured funds to other banks in the euro wholesale money market.</p>

***

***

## Overview of the dataset


```python
bank_dataset.shape
```




    (41188, 21)




```python
bank_dataset.iloc[:,:10].head(2)
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
      <th>job</th>
      <th>marital</th>
      <th>education</th>
      <th>default</th>
      <th>housing</th>
      <th>loan</th>
      <th>contact</th>
      <th>month</th>
      <th>day_of_week</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>56</td>
      <td>housemaid</td>
      <td>married</td>
      <td>basic.4y</td>
      <td>no</td>
      <td>no</td>
      <td>no</td>
      <td>telephone</td>
      <td>may</td>
      <td>mon</td>
    </tr>
    <tr>
      <th>1</th>
      <td>57</td>
      <td>services</td>
      <td>married</td>
      <td>high.school</td>
      <td>unknown</td>
      <td>no</td>
      <td>no</td>
      <td>telephone</td>
      <td>may</td>
      <td>mon</td>
    </tr>
  </tbody>
</table>
</div>




```python
bank_dataset.iloc[:,10:].head(2)
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
      <th>duration</th>
      <th>campaign</th>
      <th>pdays</th>
      <th>previous</th>
      <th>poutcome</th>
      <th>emp.var.rate</th>
      <th>cons.price.idx</th>
      <th>cons.conf.idx</th>
      <th>euribor3m</th>
      <th>nr.employed</th>
      <th>y</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>261</td>
      <td>1</td>
      <td>999</td>
      <td>0</td>
      <td>nonexistent</td>
      <td>1.1</td>
      <td>93.994</td>
      <td>-36.4</td>
      <td>4.857</td>
      <td>5191.0</td>
      <td>no</td>
    </tr>
    <tr>
      <th>1</th>
      <td>149</td>
      <td>1</td>
      <td>999</td>
      <td>0</td>
      <td>nonexistent</td>
      <td>1.1</td>
      <td>93.994</td>
      <td>-36.4</td>
      <td>4.857</td>
      <td>5191.0</td>
      <td>no</td>
    </tr>
  </tbody>
</table>
</div>




```python
bank_dataset.info()
```

    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 41188 entries, 0 to 41187
    Data columns (total 21 columns):
    age               41188 non-null int64
    job               41188 non-null object
    marital           41188 non-null object
    education         41188 non-null object
    default           41188 non-null object
    housing           41188 non-null object
    loan              41188 non-null object
    contact           41188 non-null object
    month             41188 non-null object
    day_of_week       41188 non-null object
    duration          41188 non-null int64
    campaign          41188 non-null int64
    pdays             41188 non-null int64
    previous          41188 non-null int64
    poutcome          41188 non-null object
    emp.var.rate      41188 non-null float64
    cons.price.idx    41188 non-null float64
    cons.conf.idx     41188 non-null float64
    euribor3m         41188 non-null float64
    nr.employed       41188 non-null float64
    y                 41188 non-null object
    dtypes: float64(5), int64(5), object(11)
    memory usage: 6.6+ MB


> __Note:__ 
><br> There are no null values reflected above. The data types for all the columns are appropriate too.
><br> However, from the _data dictionary_:
<p>&#9679; the 2nd to 7th variable contain 'unknown' values.
<br>&#9679; the duration column also contain '0' values which may add bias to the model because a phonecall duration of 0 would immediately mean that the customer has yet to subscribe.
<br>&#9679; the days_passed column contain 39,661 values which are '999'.
<br>
<br> Hence, we need to analyse each of these columns individually to see if their values need to be addressed.</p>

***

## Amend column names


```python
new_colnames = ['age','occupation','marital','education','default_credit','housing_loan','personal_loan','contact','month','day','duration','contact_freq','days_passed','contact_bef','prev_outcome','emp_var_rate','cpi_index','cci_index','e3m','employees','subscription']
```


```python
bank_dataset.columns = new_colnames
```

***

## Identify & drop duplicates


```python
bank_dataset.duplicated().value_counts()
```




    False    41176
    True        12
    dtype: int64




```python
bank_dataset = bank_dataset.drop_duplicates()
```


```python
bank_dataset.duplicated().value_counts()
```




    False    41176
    dtype: int64



***

## Individual assessment of each column


```python
bank_dataset['age'].unique()
# no zeroes present.
```




    array([56, 57, 37, 40, 45, 59, 41, 24, 25, 29, 35, 54, 46, 50, 39, 30, 55,
           49, 34, 52, 58, 32, 38, 44, 42, 60, 53, 47, 51, 48, 33, 31, 43, 36,
           28, 27, 26, 22, 23, 20, 21, 61, 19, 18, 70, 66, 76, 67, 73, 88, 95,
           77, 68, 75, 63, 80, 62, 65, 72, 82, 64, 71, 69, 78, 85, 79, 83, 81,
           74, 17, 87, 91, 86, 98, 94, 84, 92, 89])




```python
bank_dataset['occupation'].value_counts()
```




    admin.           10419
    blue-collar       9253
    technician        6739
    services          3967
    management        2924
    retired           1718
    entrepreneur      1456
    self-employed     1421
    housemaid         1060
    unemployed        1014
    student            875
    unknown            330
    Name: occupation, dtype: int64




```python
bank_dataset['marital'].value_counts()
```




    married     24921
    single      11564
    divorced     4611
    unknown        80
    Name: marital, dtype: int64




```python
bank_dataset['education'].value_counts()
# Too many 'unknown' values here which will not provide any value to the model.
```




    university.degree      12164
    high.school             9512
    basic.9y                6045
    professional.course     5240
    basic.4y                4176
    basic.6y                2291
    unknown                 1730
    illiterate                18
    Name: education, dtype: int64




```python
bank_dataset['default_credit'].value_counts()
# Given that there is plenty of ambiguity in 8596 'unknown' values
# and that there are only 3 'yes' values, this column will be dropped later.
```




    no         32577
    unknown     8596
    yes            3
    Name: default_credit, dtype: int64




```python
bank_dataset['housing_loan'].value_counts()
```




    yes        21571
    no         18615
    unknown      990
    Name: housing_loan, dtype: int64




```python
bank_dataset['personal_loan'].value_counts()
```




    no         33938
    yes         6248
    unknown      990
    Name: personal_loan, dtype: int64




```python
# There are no issues with variables 'contact','month','day'
```


```python
bank_dataset[bank_dataset['duration'] == 0]['duration']
```




    6251     0
    23031    0
    28063    0
    33015    0
    Name: duration, dtype: int64




```python
# 'contact_freq','days_passed','contact_bef','prev_outcome'
```


```python
bank_dataset['days_passed'].value_counts()
```




    999    39661
    3        439
    6        412
    4        118
    9         64
    2         61
    7         60
    12        58
    10        52
    5         46
    13        36
    11        28
    1         26
    15        24
    14        20
    8         18
    0         15
    16        11
    17         8
    18         7
    19         3
    22         3
    21         2
    26         1
    20         1
    25         1
    27         1
    Name: days_passed, dtype: int64




```python
bank_dataset['contact_bef'].value_counts()
```




    0    35551
    1     4561
    2      754
    3      216
    4       70
    5       18
    6        5
    7        1
    Name: contact_bef, dtype: int64




```python
bank_dataset['prev_outcome'].value_counts()
```




    nonexistent    35551
    failure         4252
    success         1373
    Name: prev_outcome, dtype: int64




```python
# 'emp_var_rate','cpi_index','cci_index','e3m','employees'
```


```python
bank_dataset['emp_var_rate'].value_counts()
bank_dataset['cci_index'].value_counts()
# these have negative values in them.
```




    -36.4    7762
    -42.7    6681
    -46.2    5793
    -36.1    5173
    -41.8    4374
    -42.0    3615
    -47.1    2457
    -31.4     770
    -40.8     715
    -26.9     446
    -30.1     357
    -40.3     311
    -37.5     303
    -50.0     282
    -29.8     267
    -34.8     264
    -38.3     233
    -39.8     229
    -40.0     212
    -49.5     204
    -33.6     177
    -34.6     174
    -33.0     172
    -50.8     128
    -40.4      67
    -45.9      10
    Name: cci_index, dtype: int64




```python
# Target variable
```


```python
bank_dataset['subscription'].value_counts()
```




    no     36537
    yes     4639
    Name: subscription, dtype: int64



> __Assessment:__
1. for columns ranging from occupation to personal_loan, replace 'unknown' values with np.nan and dropna.
2. for duration column, replace '0' values with np.nan and dropna.
3. Feature engineer the column, days_passed, due to the large number of '999' values reflecting customers not being contacted for a previous campaign.
<br>- This will be feature engineered to a column describing, whether or not a customer was previously contacted for a previous campaign.
<br>- the original column will not be dropped yet as it may be useful for the EDA process. It will only be dropped just before Machine Learning Modelling.
4. drop default_credit column
<br>- Given that there is plenty of ambiguity in 8596 'unknown' values & only 3 'yes' values.
5. ignore negative values in emp_var_rate & cci_index columns for now.
6. keep in mind that the target variable 'subscription' is severely imbalanced and will need to be addressed when building Machine Learning models.

***

## Cleaning the columns


```python
# 'unknown' values replaced with np.nan for columns ranging from occupation to personal_loan.
for cols in bank_dataset.columns:
    bank_dataset[cols] = bank_dataset[cols].map(lambda x: np.nan if x == 'unknown' else x)
```


```python
# '0' values replaced with np.nan for duration column.
bank_dataset['duration'] = bank_dataset['duration'].map(lambda x: np.nan if x == 0 else x)
```


```python
# drop nan values for these respective columns.
bank_dataset = bank_dataset.dropna()
```


```python
# Feature engineer days_passed column to a column describing previous participation in another campaign or not.
bank_dataset['prev_part'] = bank_dataset['days_passed'].map(lambda x: 0 if x == 999 else 1)
```


```python
# drop default_credit column.
bank_dataset = bank_dataset.drop(labels=['default_credit'],axis=1)
```


```python
# encode subscription column with 0 or 1.
bank_dataset['subscription'] = bank_dataset['subscription'].map(lambda x: 0 if x == 'no' else 1)
```


```python
bank_dataset.head(2)
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
      <th>occupation</th>
      <th>marital</th>
      <th>education</th>
      <th>housing_loan</th>
      <th>personal_loan</th>
      <th>contact</th>
      <th>month</th>
      <th>day</th>
      <th>duration</th>
      <th>...</th>
      <th>days_passed</th>
      <th>contact_bef</th>
      <th>prev_outcome</th>
      <th>emp_var_rate</th>
      <th>cpi_index</th>
      <th>cci_index</th>
      <th>e3m</th>
      <th>employees</th>
      <th>subscription</th>
      <th>prev_part</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>56</td>
      <td>housemaid</td>
      <td>married</td>
      <td>basic.4y</td>
      <td>no</td>
      <td>no</td>
      <td>telephone</td>
      <td>may</td>
      <td>mon</td>
      <td>261.0</td>
      <td>...</td>
      <td>999</td>
      <td>0</td>
      <td>nonexistent</td>
      <td>1.1</td>
      <td>93.994</td>
      <td>-36.4</td>
      <td>4.857</td>
      <td>5191.0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>37</td>
      <td>services</td>
      <td>married</td>
      <td>high.school</td>
      <td>yes</td>
      <td>no</td>
      <td>telephone</td>
      <td>may</td>
      <td>mon</td>
      <td>226.0</td>
      <td>...</td>
      <td>999</td>
      <td>0</td>
      <td>nonexistent</td>
      <td>1.1</td>
      <td>93.994</td>
      <td>-36.4</td>
      <td>4.857</td>
      <td>5191.0</td>
      <td>0</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
<p>2 rows × 21 columns</p>
</div>




```python
bank_dataset.to_csv('bank dataset (cleaned).csv')
```
