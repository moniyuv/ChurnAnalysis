
# coding: utf-8

# In[1]:


import pandas as pd
import os
import numpy as np
import matplotlib.pyplot as plt


# In[2]:


os.chdir('C:\\Users\\bipul\\PycharmProjects\\ML\\Assignments\\Project\\data')


# In[218]:


train = pd.read_csv('training_v2.csv')
train.info()


# In[219]:


train['Days_to_expiration'] = pd.to_timedelta(train['Days_to_expiration'])
train['Days_to_expiration'] = train['Days_to_expiration'].dt.days


# In[220]:


train['Lastdayused'] = pd.to_timedelta(train['Lastdayused'])
train['Lastdayused'] = train['Lastdayused'].dt.days


# In[221]:


train.info()


# In[222]:


test = pd.read_csv('test.csv')
test.info()


# In[223]:


test['Days_to_expiration'] = pd.to_timedelta(test['Days_to_expiration'])
test['Days_to_expiration'] = test['Days_to_expiration'].dt.days


# In[224]:


test['Lastdayused'] = pd.to_timedelta(train['Lastdayused'])
test['Lastdayused'] = test['Lastdayused'].dt.days


# In[225]:


test.info()
test.groupby('is_churn').count()
test = test.dropna()


# # PCA 

# In[28]:


from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

cols = set(train.columns)
cols.remove('msno')
cols.remove('is_churn')
scale_df = train[list(cols)]

scaler = StandardScaler()
scaled = scaler.fit_transform(scale_df)

pca = PCA()
pca.fit(scaled)

pca_df = pca.transform(scaled)

per_var = np.round(pca.explained_variance_ratio_* 100, decimals=1)
labels = ['PC' + str(x) for x in range(1, len(per_var)+1)]
 
plt.bar(x=range(1,len(per_var)+1), height=per_var, tick_label=labels)
plt.ylabel('Percentage of Explained Variance')
plt.xlabel('Principal Component')
plt.title('Plot')
fig = plt.gcf()
plt.figure(figsize=(200,200))
plt.show()


# In[12]:


# pca_rec = pca.inverse_transform(pca_df)
# pca_rec


# In[13]:


# type(pca_rec)


# # Modelling

# In[57]:


import pandas
import matplotlib.pyplot as plt
from sklearn import model_selection
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
import warnings
warnings.filterwarnings('ignore')


# In[125]:


# X_train = train[list(cols)]
# y_train = train['is_churn']
validation_size = 0.20
seed = 4
scoring = 'roc_auc'

print("Train test split")

X_train, X_test, y_train, y_test = model_selection.train_test_split(train[list(cols)], 
                                                                    train['is_churn'], 
                                                                    test_size = validation_size, 
                                                                    random_state = seed)


# In[59]:


print("start preparing models ")

# prepare models
models = []
# models.append(('LR', LogisticRegression()))
models.append(('LDA', LinearDiscriminantAnalysis()))
models.append(('KNN', KNeighborsClassifier()))
models.append(('DTC', DecisionTreeClassifier()))
models.append(('NB', GaussianNB()))
models.append(('NN', MLPClassifier()))
models.append(('RFC', RandomForestClassifier()))
# models.append(('SVM', SVC()))
# evaluate each model in turn
results = []
names = []
for name, model in models:
    print("Prepare ", name)
    kfold = model_selection.KFold(n_splits=5, random_state=seed, shuffle=True)
    cv_results = model_selection.cross_val_score(model, X_train, y_train, cv=kfold, scoring=scoring)
    results.append(cv_results)
    names.append(name)
    msg = "%s: %f (%f)" % (name, cv_results.mean(), cv_results.std())
    print(msg)
# boxplot algorithm comparison
fig = plt.figure()
fig.suptitle('Algorithm Comparison')
ax = fig.add_subplot(111)
plt.boxplot(results)
ax.set_xticklabels(names)
plt.show()


# In[60]:


import random

def some(x, n):
    return x.ix[random.sample(list(x.index), n)]


# In[61]:


from sklearn import metrics
import matplotlib.pyplot as plt
import numpy as np

print("Start error metrics")

# X_test = test[list(cols)]
# y_test = test['is_churn']

for name, model in models:
    print()
    print("=================Metrics for:", name, "=================")
    print()
    probas = model.fit(X_train, y_train).predict_proba(X_test)
    y_pred = model.predict(X_test)
    print("Model accuracy:", metrics.accuracy_score(y_test, y_pred))
    print()
    print("Confusion Matrix:\n", metrics.confusion_matrix(y_test, y_pred))
    print()
    print("Classification report:\n", metrics.classification_report(y_test, y_pred))
    print()
    print("Hamming Loss:", metrics.hamming_loss(y_test, y_pred))
    print()
    print("Jaccard similarity coefficient score:", metrics.jaccard_similarity_score(y_test, y_pred))
    print()
    print("Log loss:", metrics.log_loss(y_test, y_pred))
    print()
    
    fpr, tpr, thresholds = metrics.roc_curve(y_test, probas[:, 1])
    
    x = fpr# false_positive_rate
    y = tpr# true_positive_rate 

    # This is the ROC curve
    plt.plot(x,y)
    plt.show() 

    # This is the AUC
    auc = np.trapz(y,x)
    print("Area under curve:", auc)
    print()


# In[66]:


column_sum = 0
for row in probas:
    column_sum += row[0]
print(column_sum)


# # LGB

# In[30]:


import lightgbm as lgb

lgb_params = { 'learning_rate': 0.01, 
               'application': 'binary', 
               'max_depth': 40, 
               'num_leaves': 3400, 
               'verbosity': -1, 
               'metric': 'binary_logloss' 
              } 

d_trainl = lgb.Dataset(X_train, label = y_train) 
d_validl = lgb.Dataset(X_test, label = y_test) 
watchlistl = [d_trainl, d_validl]

lgb_model = lgb.train(lgb_params, 
                      train_set = d_trainl, 
                      num_boost_round = 500, 
                      valid_sets = watchlistl, 
                      early_stopping_rounds = 50, 
                      verbose_eval = 10)

lgb_test = test[list(cols)]
lgb_actual = test['is_churn']

# lgb_probas = lgb_model.predict_proba(lgb_test)


# In[25]:


train.groupby('is_churn').count()


# In[26]:


train.count()


# In[92]:


train.info()


# In[226]:


lgb_test = some(train, 4179)
temp = some(test, 90000)
lgb_test = lgb_test.append(temp)
lgb_actual = lgb_test['is_churn']
lgb_pred = lgb_model.predict(lgb_test[list(cols)])
lgb_test['is_churn'] = lgb_pred
data = lgb_test['is_churn']
lgb_proba = data


# In[227]:


data.mean()


# In[228]:


data.median()


# In[229]:


data[data>0.3].count()


# In[230]:


data.head(5)


# In[231]:


mask = data > 0.3
data.loc[mask] = 1


# In[232]:


mask = data < 0.3
data.loc[mask] = 0


# In[233]:


data.head(3)


# In[234]:


from sklearn import metrics
import matplotlib.pyplot as plt
import numpy as np

print("=============Metrics for LGBM================")
print()
lgb_pred = data
print("Model accuracy:", metrics.accuracy_score(lgb_actual, lgb_pred))
print()
print("Confusion Matrix:\n", metrics.confusion_matrix(lgb_actual, lgb_pred))
print()
print("Classification report:\n", metrics.classification_report(lgb_actual, lgb_pred))
print()
print("Hamming Loss:", metrics.hamming_loss(lgb_actual, lgb_pred))
print()
print("Jaccard similarity coefficient score:", metrics.jaccard_similarity_score(lgb_actual, lgb_pred))
print()
print("Log loss:", metrics.log_loss(lgb_actual, lgb_pred))
print()

fpr, tpr, thresholds = metrics.roc_curve(lgb_actual, lgb_proba)
    
x = fpr# false_positive_rate
y = tpr# true_positive_rate 

# This is the ROC curve
plt.plot(x,y)
plt.show() 

# This is the AUC
auc = np.trapz(y,x)
print("Area under curve:", auc)
print()


# In[216]:


temp.info()


# In[217]:


lgb_test.info()

