
# coding: utf-8

# In[1]:

import pandas as pd # pandas
import numpy as np
import matplotlib.pyplot as plt # module for plotting
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

data = [pd.read_csv('http://api.qdatum.io/v1/pull/' + str(i) +'?format=tsv', sep='\t') for i in range(1, 17)]


# In[3]:

time_series = data[1].copy()
del time_series['pos'] # remove useless columns that prevent duplicates to be identified
del time_series['link']
time_series = time_series[(time_series.value == time_series.value) & (time_series.value != ' ')] # remove NaN values
time_series.value = time_series.value.astype(int) # convert values from string to int
time_series = time_series.drop_duplicates() # remove duplicates


# In[4]:

#Standardize source name
def normalize_source(source):
    source = source.upper()
    if source == 'WHO; GVT': source = 'WHO'
    return source

time_series.sources = time_series.sources.apply(normalize_source)


# In[5]:

# temporary, should look into the details
time_series.sdr_level = time_series.sdr_level.fillna('national')


# In[7]:

time_series_list = []
_d = time_series.to_dict()
for i in time_series.index:
    d = {}
    for c in time_series.columns:
        d[c] = _d[c][i]
    time_series_list.append(d)

# use a dictionary to clean data
# create the dictionary
time_series_dict = {}
for i, el in enumerate(time_series_list):
    loc = el['country_code'] + str(el['sdr_id']) + el['sdr_level']
    if loc not in time_series_dict: time_series_dict[loc] = {}
    if el['date'] not in time_series_dict[loc]: time_series_dict[loc][el['date']] = []
    time_series_dict[loc][el['date']].append(el)


def clean(reported_cases, past_cases):
    """
    reported cases: list of cases reported for a given location (of all categories)
    past_cases: dict of categories and past values
    """
    cases = {c: {} for c in past_cases}
    sources_preference = ['WHO', 'GVT', 'MINISTERE DE LA SANTE', 'UNICEF', 'ECHO'] # ordered by our preference in sources
    for el in reported_cases:
        if el['category'] == 'New cases' or el['value'] >= past_cases[el['category']]: # make sure the values are increasing
            if not el['sources'] in cases[el['category']]: cases[el['category']][el['sources']] = []
            cases[el['category']][el['sources']].append(el)
    for c in cases:
        if len(cases[c]):
            for s in sources_preference: # select values for the source of preference
                if s in cases[c]:
                    m = np.array([x['value'] for x in cases[c][s]]).mean()
                    cases[c] = cases[c][s][0]
                    cases[c]['value'] = m # and average over duplicate values if any
                    break
    cases = {c: cases[c] for c in cases if len(cases[c])}
    return cases
    
clean_data = []
time_series_dict2 = {loc: {c: {} for c in time_series.category.unique()} for loc in time_series_dict}
for loc in time_series_dict:
    past_cases = {c: 0 for c in time_series.category.unique()} # initiate at 0
    # for each location, sort the data by date and clean it
    for date in sorted(time_series_dict[loc]):
        c = clean(time_series_dict[loc][date], past_cases)
        for k in c:
            clean_data.append(c[k])
            time_series_dict2[loc][k][date] = c[k]
        for k in c: past_cases[k] = c[k]['value']
                
ts_clean = pd.DataFrame(clean_data)
print len(time_series.index), len(ts_clean.index)
ts_clean.head()


# In[8]:

# interpolate missing data
first_day = ts_clean.date.min() - 1
interpolated_data = []
for loc in time_series_dict2:
    for c in time_series_dict2[loc]:
        last_day, last_value = first_day, 0
        for d in sorted(time_series_dict2[loc][c]):
            new_value = time_series_dict2[loc][c][d]['value']
            for i in range(last_day + 1, d):
                v = int(last_value + (new_value - last_value) * (i - last_day) * 1.0 / (d - last_day))
                el = time_series_dict2[loc][c][d].copy()
                el['value'] = v
                el['type'] = 'interpolate'
                el['date'] = i
                interpolated_data.append(el)
            time_series_dict2[loc][c][d]['type'] = 'original'
            interpolated_data.append(time_series_dict2[loc][c][d])
            last_day, last_value = d, new_value

ts_interpolated = pd.DataFrame(interpolated_data)
print len(time_series.index), len(ts_interpolated.index)
ts_interpolated.head()


# In[38]:

# look at data repartition
ts_interpolated[ts_interpolated.type == 'original'].groupby(['country_code', 'sdr_id']).count()


# In[39]:

df = ts_interpolated[(ts_interpolated.country_code == 'LR') & (ts_interpolated.sdr_id == 5513)]
df[df.type == 'original'].groupby('category').count()


# In[40]:

def plot(country_code, sdr_id):
    df = ts_interpolated[(ts_interpolated.country_code == country_code) & (ts_interpolated.sdr_id == sdr_id)]
    for c in df.category.unique():
        _df = df[(df.category == c) & (df.type == 'original')]
        plt.plot(_df.date, _df.value, label=c)
    plt.legend(loc=2)
    plt.show()
plot('LR', 5513)
    


# In[ ]:



