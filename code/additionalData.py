
# coding: utf-8

# In[1]:

import pandas as pd # pandas
import numpy as np
import matplotlib.pyplot as plt # module for plotting
from numpy.linalg import inv
import statsmodels.api as sm
import datetime as dt
get_ipython().magic(u'matplotlib inline')

#nice defaults for matplotlib
from matplotlib import rcParams

dark2_colors = [(0.10588235294117647, 0.6196078431372549, 0.4666666666666667),
                (0.8509803921568627, 0.37254901960784315, 0.00784313725490196),
                (0.4588235294117647, 0.4392156862745098, 0.7019607843137254),
                (0.9058823529411765, 0.1607843137254902, 0.5411764705882353),
                (0.4, 0.6509803921568628, 0.11764705882352941),
                (0.9019607843137255, 0.6705882352941176, 0.00784313725490196),
                (0.6509803921568628, 0.4627450980392157, 0.11372549019607843),
                (0.4, 0.4, 0.4)]

rcParams['figure.figsize'] = (10, 6)
rcParams['figure.dpi'] = 150
rcParams['axes.color_cycle'] = dark2_colors
rcParams['lines.linewidth'] = 2
rcParams['axes.grid'] = True
rcParams['axes.facecolor'] = '#eeeeee'
rcParams['font.size'] = 14
rcParams['patch.edgecolor'] = 'none'


# In[2]:

GN = pd.read_csv('../data/guinea-evd-case-numbers-by-district.csv')
LB = pd.read_csv('../data/liberia-evd-case-numbers-by-district.csv')
SL = pd.read_csv('../data/sierra-leone-evd-case-numbers-by-district.csv')
cases = {'GN': GN, 'LB': LB, 'SL': SL}


# In[3]:

def week_code_to_int(code):
    year = int(code.split('-')[0])
    week = int(code.split('-W')[1])
    return 52 * (year - 2014) + week

for k in cases:
    cases[k] = cases[k][['LOCATION (CODE)', 'EPI_WEEK (CODE)', 'EBOLA_MEASURE (DISPLAY)', 'Numeric']].dropna()
    cases[k].dropna()
    cases[k]['EPI_WEEK (CODE)'] = cases[k]['EPI_WEEK (CODE)'].apply(week_code_to_int)
    


# In[4]:

cases['GN'].head(5)


# In[ ]:

for c in cases:
    df = cases[c]
    for loc in df['LOCATION (CODE)'].unique():
        w = sorted(df[df['LOCATION (CODE)'] == loc]['EPI_WEEK (CODE)'].unique())
        if len(w) > 15:
            n = [df[(df['LOCATION (CODE)'] == loc) & (df['EPI_WEEK (CODE)'] == i)].to_dict()['Numeric'].popitem()[1] for i in w]
            n = [sum(n[:i]) for i in range(len(n))]
            plt.scatter(w, n)
            plt.title(c + ' - ' + loc)
            plt.show()


# In[ ]:



