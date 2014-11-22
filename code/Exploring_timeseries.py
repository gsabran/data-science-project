
# coding: utf-8

# In[1]:

import pandas as pd # pandas
import numpy as np


# In[8]:

### Read data
data = [pd.read_csv('http://api.qdatum.io/v1/pull/' + str(i) +'?format=tsv', sep='\t') for i in range(1, 17)]
time_series = data[1]
time_series.head()


# In[139]:

### Basic exploration
print len(time_series)
for col in time_series.columns: 
    print col, len(time_series[col].value_counts())
for col in time_series.columns[1::]:
    print col, sorted(list(time_series[col].unique()))
    
#Checking sdr names
time_series.groupby(['country','sdr_name','sdr_level'], as_index = False).agg({'value' : 'count'})    

#Standardize source name
for i in range(21971):
    if time_series['sources'][i] in ['Gvt', 'gvt', 'GVT', 'WHO; Gvt']:
        time_series['sources'][i] = 'GVT'

#Checking sources: We want to check whether the sources are complementary, should we keep all? should we remove some?
time_series.groupby(['sources','category'], as_index = False).agg({'value' : 'count'})
Cases2 = time_series.groupby(['sdr_name','date','category'], as_index = False).agg({'value' : 'count'})
Cases2[Cases2.value == 2]
time_series[(time_series.sdr_name == 'Yomou') & (time_series.date == 41950 )]  
time_series2 = time_series[time_series.sources == 'GVT']
len(time_series2)    

###Organizing the data with a different structure
cross_table = pd.pivot_table(time_series2, values='value', index=['country','sdr_name', 'date', 'country_code','sdr_id','sdr_level'], columns=['category'], aggfunc=np.sum)
cross_table.reset_index(inplace=True) 
cross_table = cross_table[['country','sdr_name', 'date', 'Cases', 'Confirmed cases', 
                          'Probable cases', 'Suspected cases', 'New cases', 'Deaths', 
                           'country_code', 'sdr_id', 'sdr_level']]
cross_table.head(n=10)

