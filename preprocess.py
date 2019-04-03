
# coding: utf-8

# CHURN ANALYSIS

# ### Import Libraries

# In[ ]:


import os 
import math 
import numpy as np 
import pandas as pd 
get_ipython().magic('matplotlib inline')
import matplotlib.pyplot as plt 
import scipy as sp 
import seaborn as sns
import time
from datetime import datetime
from collections import Counter
from subprocess import check_output


# ### Read Data

# In[2]:


os.chdir('C:\\Users\\moniy\\Desktop\\Churn\\data')


# In[3]:


train = pd.read_csv("train.csv", dtype = {'is_churn': 'int8'})
train.info()
train.head(5)


# In[4]:


test = pd.read_csv("sample_submission_zero.csv", dtype = {'is_churn': 'int8'})
test.info()


# In[5]:


print(train['is_churn'].unique())                          ##Unique values in the column
no_of_churners = train['is_churn'].sum()                   ##Total number of people who churned
total_users = train['is_churn'].count()                    ## Total user base
perc_churners = (no_of_churners/total_users)*100           ## % of user base that churned
print(perc_churners)

train['is_churn'].value_counts().plot(kind = 'bar')       ##Plot pf active subscribers versus churned users


# In[6]:


userlogs = pd.read_csv("user_logs.csv", nrows = 1000000)
userlogs_2 = pd.read_csv("user_logs_v2.csv",nrows = 1000000)
userlogs = userlogs.append(userlogs_2, ignore_index = True)
userlogs.info()
del userlogs_2


# In[7]:


userlogs.head(3)


# In[8]:


members = pd.read_csv("members_v3.csv")
members.info()
members.head(5)


# In[56]:


transactions = pd.read_csv("transactions.csv")
transactions_2 = pd.read_csv("transactions_v2.csv")
transactions = transactions.append(transactions_2, ignore_index = True)
del transactions_2


# In[53]:


transactions.info()


# ### Missing values treatment

# In[15]:


members.isnull().sum()/len(members)*100                         ##% of missing values in members


# In[9]:


## There are no missing values except in gender at 65%.The missing values were replaced with "Others"
members['gender'] = members['gender'].fillna("others")


# In[17]:


members['gender'].unique()


# In[54]:


transactions.isnull().sum()/len(transactions)*100               ##% of missing values in transactions


# In[19]:


train.isnull().sum()/len(train)*100


# In[20]:


test.isnull().sum()/len(test)*100


# In[21]:


userlogs.isnull().sum()/len(userlogs)*100


# ### Outlier treatment 

# In[70]:


transactions.describe()


# In[22]:


userlogs.describe()


# In[23]:


members.describe()


# In[10]:


## Birthday has a lot of outliers; Hence deleting the column
members = members.drop('bd', axis = 1)


# In[25]:


members.head(5)


# ### Creating Analytical Dataset

# Have to create a user(msno) level dataset. Therefore the userlogs dataset has to be rolled up to user level from its current user X date level

# In[11]:


userlogs['DateTime'] = pd.to_datetime(userlogs['date'], format='%Y%m%d')


# In[12]:


userlogs['Rank'] = userlogs.groupby(['msno'])['DateTime'].rank(ascending=True)
userlogs['Rank2'] = userlogs['Rank']+1
userlogs2 = userlogs.merge(userlogs, left_on = ['msno', 'Rank'], right_on = ['msno', 'Rank2'], how = 'left')


# In[13]:


userlogs2['TimeBetweenLogins']= (userlogs2['DateTime_x'] - userlogs2['DateTime_y']).dt.days


# In[14]:


userlogs2 = userlogs2[userlogs2['TimeBetweenLogins'].notnull()]


# In[15]:


userlogs2.head(3)


# In[16]:


userlogs2 = userlogs2.groupby('msno')['TimeBetweenLogins'].agg(['sum', 'count'])
userlogs2['AvgDaysBetweenLogins'] = userlogs2['sum']/userlogs2['count']
userlogs2 = userlogs2.reset_index()


# In[17]:


list(userlogs2.columns.values)


# In[18]:


userlogs['no_of_day']=1


# In[19]:


userlogs = userlogs.groupby('msno').agg({'num_25': [np.sum,np.mean], 'num_50': [np.sum, np.mean],'num_75':[np.sum,np.mean],'num_985':[np.sum,np.mean],'num_100':[np.sum,np.mean],'num_unq':[np.sum,np.mean],'total_secs':[np.sum,np.mean],'no_of_day':[np.sum],'DateTime':[np.max]}) 


# In[20]:


userlogs.columns = ["_".join(x) for x in userlogs.columns.ravel()]


# In[21]:


userlogs.head(3)


# In[25]:


userlogs = userlogs.reset_index()


# In[22]:


userlogs['secs_per_song'] = userlogs['total_secs_sum'].div(userlogs['num_25_sum'] + userlogs['num_50_sum'] + userlogs['num_75_sum'] + userlogs['num_985_sum'] + userlogs['num_100_sum'])


# In[28]:


userlogs.shape


# In[27]:


userlogs['msno'].unique().shape


# In[29]:


userlogs_final = userlogs.merge(userlogs2, right_on='msno', left_index=True, how='left')


# In[30]:


userlogs_final = userlogs_final.drop(['sum','count'],axis=1)


# In[31]:


userlogs_final['AvgDaysBetweenLogins'].fillna(0 , inplace = True)


# In[32]:


del userlogs2


# In[33]:


userlogs_final.head(5)


# In[34]:


del userlogs


# In[37]:


userlogs_final['msno_x'].unique().shape


# In[38]:


userlogs_final.shape


# In[40]:


userlogs_final = userlogs_final.drop(['msno_y','msno'],axis=1)


# In[74]:


userlogs_final = userlogs_final.rename(columns = {'msno_x':'msno'})


# ### Transaction Dataset cleaning and roll up
Have to create a user(msno) level dataset. Therefore the transactions dataset has to be rolled up to user level from its current user X transaction level
# In[43]:


transactions.info()


# In[44]:


sns.countplot(x = "payment_plan_days", data = transactions)


# In[57]:


# as payment_plan_days has very low variance, we removed payment_plan_days
del transactions['payment_plan_days']


# In[58]:


del transactions['plan_list_price'] ## as the amount paid always equals the list price


# In[59]:


transactions['membership_expire_date'] = pd.to_datetime(transactions['membership_expire_date'], format='%Y%m%d')


# In[60]:


transactions['transaction_date'] = pd.to_datetime(transactions['transaction_date'], format='%Y%m%d')


# In[61]:


transactions = transactions.groupby('msno').agg({'payment_method_id':['nunique'],'actual_amount_paid':[np.sum],'is_auto_renew':[np.mean],'is_cancel':[np.mean],'transaction_date':[np.max,'count'],'membership_expire_date':[np.max]}) 


# In[68]:


transactions.head(5)


# In[63]:


transactions.columns = ["_".join(x) for x in transactions.columns.ravel()]


# In[65]:


transactions = transactions.reset_index()


# In[66]:


transactions.shape


# In[67]:


transactions['msno'].unique().shape


# ### Join all datasets

# In[69]:


#encoding
gender = {'male': 0, 'female': 1, 'others' :2}


# In[70]:


training = pd.merge(left = train, right = members, how = 'left', on = ['msno'])


# In[71]:


training = pd.merge(left = training, right = transactions , how = 'left', on = ['msno'])


# In[75]:


training = pd.merge(left = training, right = userlogs_final, how = 'left', on = ['msno'])


# In[76]:


training['gender'] = training['gender'].map(gender)


# In[77]:


training.shape


# In[78]:


training = training.dropna(how='any')


# In[79]:


training.info()


# In[80]:


training['msno'].unique().shape


# In[81]:


testing = pd.merge(left = test, right = members, how = 'left', on = ['msno'])


# In[82]:


del members


# In[83]:


testing = pd.merge(left = testing, right = transactions , how = 'left', on = ['msno'])


# In[84]:


del transactions


# In[85]:


testing = pd.merge(left = testing, right = userlogs_final, how = 'left', on = ['msno'])


# In[86]:


del userlogs_final


# In[87]:


testing['gender'] = testing['gender'].map(gender)


# In[88]:


testing.shape


# In[89]:


testing = testing.dropna(how='any')


# In[90]:


testing.info()


# In[91]:


testing['msno'].unique().shape


# ### Churn Analysis

# ### Bivariate plots

# #### By Gender

# In[92]:


gender = training.groupby(['is_churn', 'gender']).agg({'msno': 'count'})


# In[93]:


gender.head(6)


# In[94]:


gender_pcts = gender.groupby(level=0).apply(lambda x:100 * x / float(x.sum()))


# In[95]:


gender_pcts = gender_pcts.reset_index()


# In[96]:


gender_pcts


# In[98]:


barwidth = 0.25

active = [27.35,25.03,47.61]
churners = [36.85,32.08,31.05]
 
# Set position of bar on X axis
r1 = np.arange(3)
r2 = [x + barwidth for x in r1]
r3 = [x + barwidth for x in r2]
 
plt.bar(r1, active,width = barwidth,label='Active')
plt.bar(r2, churners,width = barwidth,label='Churned')
plt.xticks([r + (barwidth*0.5) for r in range(len(active))], ['Male', 'Female','Others'])
plt.legend()
plt.title('Percentage of Churners Versus Active Subscribers by Gender')
plt.savefig('GenderVsChurn.png', dpi = 300)
plt.show()

Note: This variable is important as we can see that the users that don't provide gender tend to be active subscribers and the ones who do have a higher propensity to churn 
# #### By City

# In[99]:


city = training.groupby(['is_churn', 'city']).agg({'msno': 'count'})


# In[100]:


city_pcts = city.groupby(level=0).apply(lambda x:100 * x / float(x.sum())).reset_index()


# In[101]:


active_city = city_pcts.loc[city_pcts['is_churn'] == 0]
active_city = active_city['msno']


# In[102]:


churners_city = city_pcts.loc[city_pcts['is_churn'] == 1]
churners_city = churners_city['msno']


# In[103]:


barwidth=0.2
plt.figure(figsize=(20,6))
# Set position of bar on X axis
r1 = np.arange(len(active_city))
r2 = [x + barwidth for x in r1]
r3 = [x + barwidth for x in r2]
 
plt.bar(r1, active_city,width = barwidth,label='Active')
plt.bar(r2, churners_city,width = barwidth,label='Churned')
plt.xticks([r + (barwidth*0.5) for r in range(len(churners_city))], city_pcts['city'].unique())
plt.legend()
plt.title('Percentage of Churners Versus Active Subscribers by City')
plt.savefig('CityVsChurn.png', dpi = 300)
plt.show()

Note: There is differences in behavior towards churn by city. Therefore, this can be a variable to the model
# #### By registration Method

# In[104]:


reg_method = training.groupby(['is_churn', 'registered_via']).agg({'msno': 'count'})
reg_pcts = reg_method.groupby(level=0).apply(lambda x:100 * x / float(x.sum())).reset_index()
reg_pcts


# In[105]:


active_reg = reg_pcts.loc[reg_pcts['is_churn'] == 0]
active_reg = active_reg['msno']

churners_reg = reg_pcts.loc[reg_pcts['is_churn'] == 1]
churners_reg = churners_reg['msno']


# In[106]:


barwidth=0.25
plt.figure(figsize=(10,6))
# Set position of bar on X axis
r1 = np.arange(len(active_reg))
r2 = [x + barwidth for x in r1]
r3 = [x + barwidth for x in r2]
 
plt.bar(r1, active_reg,width = barwidth,label='Active')
plt.bar(r2, churners_reg,width = barwidth,label='Churned')
plt.xticks([r + (barwidth*0.5) for r in range(len(churners_reg))], reg_pcts['registered_via'].unique())
plt.legend()
plt.title('Percentage of Churners Versus Active Subscribers by Registration Method')
plt.savefig('RegistrationMethodVsChurn.png', dpi = 300)
plt.show()

Note: There are differences in behavior here in type 7 versus the other types
# #### Registration Time

# In[107]:


last_date = datetime.strptime('20170331', "%Y%m%d").date()
training['no_days_since_registration'] = training.registration_init_time.apply(lambda x: (last_date - datetime.strptime(str(int(x)), "%Y%m%d").date()).days if pd.notnull(x) else "NAN")


# In[108]:


del training['registration_init_time']


# In[109]:


training['no_days_since_registration'].isnull().sum()


# In[110]:


testing['no_days_since_registration'] = testing.registration_init_time.apply(lambda x: (last_date - datetime.strptime(str(int(x)), "%Y%m%d").date()).days if pd.notnull(x) else "NAN")


# In[111]:


del testing['registration_init_time']


# ### Last Transaction Date

# In[120]:


training['Days_to_expiration']= training['membership_expire_date_amax'] - training['transaction_date_amax'] 
testing['Days_to_expiration']= testing['membership_expire_date_amax'] - testing['transaction_date_amax'] 


# In[116]:


training.head(3)


# In[121]:


del training['membership_expire_date_amax']
del testing['membership_expire_date_amax']


# In[118]:


training.head(3)


# In[122]:


del training['transaction_date_amax']
del testing['transaction_date_amax']


# ### Correlation

# In[123]:


plt.matshow(training.corr())


# In[124]:


corr = training.corr()
sns.heatmap(corr, 
            xticklabels=corr.columns.values,
            yticklabels=corr.columns.values)


# In[126]:


correlation_matrix = training.corr(method='pearson')


# In[128]:


correlation_matrix.to_csv('CorrelationMatrixv1.csv', encoding='utf-8', index=True)


# In[130]:


training = training.drop(['num_75_sum','num_unq_sum','num_100_mean','num_unq_mean','total_secs_sum','total_secs_mean'],axis=1)


# In[131]:


testing = testing.drop(['num_75_sum','num_unq_sum','num_100_mean','num_unq_mean','total_secs_sum','total_secs_mean'],axis=1)


# In[132]:


training.info()


# In[133]:


training['Lastdayused']= last_date- training['DateTime_amax'] 
testing['Lastdayused']= last_date- testing['DateTime_amax'] 


# In[134]:


del training['DateTime_amax'] 
del testing['DateTime_amax']


# In[135]:


training.info()


# In[136]:


training.isnull().sum()/len(training)*100


# In[137]:


testing.isnull().sum()/len(testing)*100


# In[139]:


training['msno'].unique().shape


# In[140]:


training.to_csv('training_v2.csv', encoding='utf-8', index=False)


# In[141]:


testing.to_csv('test.csv',encoding ='utf-8',index = False)

