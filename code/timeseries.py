
# coding: utf-8

# In[101]:

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


# In[102]:

data = [pd.read_csv('http://api.qdatum.io/v1/pull/' + str(i) +'?format=tsv', sep='\t') for i in range(1, 17)]


# In[103]:

time_series = data[1].copy()
del time_series['pos'] # remove useless columns that prevent duplicates to be identified
del time_series['link']
time_series = time_series[(time_series.value == time_series.value) & (time_series.value != ' ')] # remove NaN values
time_series.value = time_series.value.astype(int) # convert values from string to int
time_series = time_series.drop_duplicates() # remove duplicates


# In[104]:

#Standardize source name
def normalize_source(source):
    source = source.upper()
    if source == 'WHO; GVT': source = 'WHO'
    return source

time_series.sources = time_series.sources.apply(normalize_source)


# In[105]:

# show the different sources for Guinea
def plot_raw(country_code, sdr_id):
    df = time_series[(time_series.country_code == country_code) & (time_series.sdr_id == sdr_id) & (time_series.category == 'Cases')]
    markers = ['x', 's', 'o', 'o', 'o', 'o', 'o', 'o', 'o', 'o']
    colors = ['b', 'r', 'g']
    for idx, s in enumerate(df.sources.unique()):
        _df = df[(df.sources == s)]
        plt.scatter(_df.date, _df.value, label=s, marker = markers[idx], color = colors[idx])
        
    plt.legend(loc=2)
    plt.show()
#plt.xlim(41860, 41920)
plot_raw('GN', 0)


# In[106]:

# temporary, should look into the details
time_series.sdr_level = time_series.sdr_level.fillna('national')


# In[315]:

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
            # TODO remove this line
            #if date > 41883: time_series_dict2[loc][k][date]['value'] = 0
            #if date > 41883: time_series_dict2[loc][k][date]['delta'] = 0
        for k in c: past_cases[k] = c[k]['value']
                
ts_clean = pd.DataFrame(clean_data)
print len(time_series.index), len(ts_clean.index)
ts_clean.head()


# In[316]:

df = ts_clean[(ts_clean.date <41920) & (ts_clean.date >41900) & (ts_clean.country_code == 'GN') & (ts_clean.sdr_id == 0) & (ts_clean.category == 'Cases')]
df.head()


# In[317]:

# interpolate missing data
first_day = ts_clean.date.min() - 1
interpolated_data = []

# find day ebola is reported for the first time
ebola_start_days = {}
for loc in time_series_dict2:
    start_day = 10000000
    for c in time_series_dict2[loc]:
        if len(time_series_dict2[loc][c]):
            start_day = min(start_day, sorted(time_series_dict2[loc][c])[0])
    ebola_start_days[loc] = start_day
        
# interpolate the data
for loc in time_series_dict2:
    for c in time_series_dict2[loc]:
        last_day, last_value, yesterday_value = first_day, 0, 0
        for d in sorted(time_series_dict2[loc][c]):
            new_value = time_series_dict2[loc][c][d]['value']
            for i in range(last_day + 1, d):
                ebola_started = ebola_start_days[loc] <= i
                v = 0
                if ebola_started:  v = int(last_value + (new_value - last_value) * (i - last_day) * 1.0 / (d - last_day))
                el = time_series_dict2[loc][c][d].copy()
                el['value'] = v
                el['type'] = 'interpolate'
                el['date'] = i
                el['ebola_already_reported'] = ebola_started
                el['delta'] = v - yesterday_value
                yesterday_value = v
                interpolated_data.append(el)
                time_series_dict2[loc][c][i] = el
            ebola_started = ebola_start_days[loc] <= d
            time_series_dict2[loc][c][d]['type'] = 'original'
            time_series_dict2[loc][c][d]['ebola_already_reported'] = ebola_started
            time_series_dict2[loc][c][d]['delta'] = new_value - yesterday_value
            yesterday_value = new_value
            interpolated_data.append(time_series_dict2[loc][c][d])
            last_day, last_value = d, new_value
            
# add recent deaths
recent = 10 # 10 days
for loc in time_series_dict2:
    c = 'Deaths'
    recent_deaths = []
    for d in sorted(time_series_dict2[loc][c]):
        el = time_series_dict2[loc][c][d].copy()
        removed = 0
        if len(recent_deaths) == recent: removed = recent_deaths[recent - 1]
        recent_deaths = recent_deaths[:recent - 1]
        recent_deaths.append(el['delta'])
        el['category'] = 'Recent Deaths'
        el['value'] = sum(recent_deaths)
        el['delta'] = el['delta'] - removed
        el['type'] = 'interpolate'
        # TODO remove this line
        #if d > 41883: el['value'] = 0
        #if d > 41883: el['delta'] = 0
        interpolated_data.append(el)


ts_interpolated = pd.DataFrame(interpolated_data)
print len(time_series.index), len(ts_interpolated.index)
ts_interpolated.head()


# In[318]:

# look at data repartition
ts_interpolated[ts_interpolated.type == 'original'].groupby(['country_code', 'sdr_id']).count().head()


# In[319]:

df = ts_interpolated[(ts_interpolated.country_code == 'LR') & (ts_interpolated.sdr_id == 5513)]
df[df.type == 'original'].groupby('category').count()


# In[320]:

def plot(country_code, sdr_id, categories=ts_interpolated.category.unique()):
    df = ts_interpolated[(ts_interpolated.country_code == country_code) & (ts_interpolated.sdr_id == sdr_id)]
    for c in categories:
        _df = df[(df.category == c)]
        plt.plot(_df.date, _df.value, label=c)
        
    plt.legend(loc=2)
#plt.xlim(41860, 41920)
plot('GN', 0)
plt.show()


# In[425]:

a = None
def get_first(df, k):
    # return first value in column k
    return df.to_dict()[k].popitem()[1]

def to_X_Y(country_code, sdr_id):
    """convert data to OLS matrixes"""
    df = ts_interpolated[(ts_interpolated.country_code == country_code) & (ts_interpolated.sdr_id == sdr_id)]
    X, Y, dates = [], [], []
    for d in df.date.unique():
        _df = df[df.date == d]
        infected = _df[_df.category == 'Cases']
        if get_first(infected, 'ebola_already_reported'):
            recent_deaths = _df[_df.category == 'Recent Deaths']
            deaths = _df[_df.category == 'Deaths']
            I = get_first(infected, 'value')
            #print d
            D_recent = get_first(recent_deaths, 'value')
            
            y = get_first(infected, 'delta')
            x = [I, -I, 0]
            X.append(x)
            Y.append([y])
            
            y = get_first(deaths, 'delta')
            x = [0, I, 0]
            X.append(x)
            Y.append([y])
            dates.append(d)
    return np.array(X), np.array(Y), dates

def fit_data(country_code, sdr_id, train_limit, days_after, should_print_table):
    train_limit = (dt.datetime.strptime(train_limit, '%Y-%m-%d') - dt.datetime(1899, 12, 30)).days
    print 'train_limit', train_limit
    # convert the data to matrix
    X, Y, dates = to_X_Y(country_code, sdr_id)
    train_limit_range = train_limit - dates[0]
    # remove the beginning of the epidemy
    X_truncated = X[days_after * 2:, :]
    Y_truncated = Y[days_after * 2:, :]
    # remove the non training data
    X_truncated_train = X_truncated[:(train_limit_range - days_after) * 2, :]
    if X_truncated_train.shape[0] == 0: return None # not any data to fit
    Y_truncated_train = Y_truncated[:(train_limit_range - days_after) * 2, :]
    # fit the OLS
    model = sm.OLS(Y_truncated_train, X_truncated_train)
    results = model.fit()
    # print the OLS stats
    if should_print_table: print(results.summary())
    
    # build the fitted times series step by step
    # get the last known data
    fit = X_truncated_train.dot(results.params)
    last_data_known = X_truncated_train[-2:, :]
    
    recent_deaths = []
    df = ts_interpolated[(ts_interpolated.country_code == country_code) & (ts_interpolated.sdr_id == sdr_id) & (ts_interpolated.category == 'Deaths')]
    for d in range(train_limit - 10, train_limit):
        _df = df[df.date == d]
        recent_deaths.append(get_first(_df, 'delta'))
    
    
    # build the predicted data from the estimated rates and the past estimated data
    for i in range(train_limit_range, len(Y) / 2):
        data_interpolated = last_data_known.dot(results.params)
        fit = np.append(fit, data_interpolated)
        delta_I = data_interpolated[0]
        delta_D = data_interpolated[1]
        recent_deaths = recent_deaths[1:]
        recent_deaths.append(delta_D)
        last_data_known = last_data_known + np.array([[delta_I, -delta_I, 0], [0, delta_I, 0]])
        #last_data_known[1, 2] = sum(recent_deaths)
    
    cum_y, cum_d_recent = [], []
    y, d_recent = sum(Y[:days_after * 2:2]), sum(Y[1:days_after * 2 + 1:2])
    if type(y) != int:
        y, d_recent = y[0], d_recent[0]
    for idx, v in enumerate(fit):
        if idx % 2 == 0:
            y += v
            cum_y.append(y)
        else:
            d_recent += v
            cum_d_recent.append(d_recent)
            
    return cum_y, cum_d_recent, dates[days_after:]

def plot_fit(country_code, sdr_id, train_limit='2014-10-01', days_after=0, should_print_table=False):
    fit = fit_data(country_code, sdr_id, train_limit, days_after, should_print_table)
    if fit is None: return
    Y_hat, D_hat, dates = fit
    plt.plot(dates, Y_hat, label='Cases fit')
    plt.plot(dates, D_hat, label='Deaths fit')
    
    # plot trianing limit
    train_limit = (dt.datetime.strptime(train_limit, '%Y-%m-%d') - dt.datetime(1899, 12, 30)).days
    plot(country_code, sdr_id, ['Cases', 'Deaths'])
    ax = plt.axis()
    plt.plot([train_limit, train_limit], [0, 1e50], color='r', label='training limit')
    plt.axis(ax)
    
    plt.legend(loc=2)


# In[426]:

country_code, sdr_id = 'GN', 0
#plot_fit(country_code, sdr_id, train_limit='2014-09-01', days_after=0)
#plt.show()
plot_fit(country_code, sdr_id, train_limit='2014-09-01', days_after=10, should_print_table=True)


# In[408]:

country_code, sdr_id = 'SL', 0
plot_fit(country_code, sdr_id, train_limit='2014-10-01', days_after=40, should_print_table=True)


# In[409]:

plot('SL', 0)


# In[410]:

selected_regions = pd.read_csv('Selected_SDR.csv', index_col =0)
selected_regions.head()


# In[411]:

GN = pd.read_csv('Guinea_artificial.csv', index_col =0)
GN.head()


# In[429]:

covariates = pd.read_csv('../data/merged_covariate_df.csv', index_col =0)
covariates.head(100)


# In[412]:

country_code, sdr_id = 'SL', 8624 
df = ts_interpolated[(ts_interpolated.country_code == country_code) & (ts_interpolated.sdr_id == sdr_id)]
_df = df[df.date == 41934]
recent_deaths = _df[_df.category == 'Recent Deaths']
df[(df.category == 'Recent Deaths') & (df.date > 41930)].head(100)


# In[424]:

for el in selected_regions.as_matrix():
    try:
        country_code, sdr_id = el
        if country_code != 'GN':
            print country_code, sdr_id
            df = ts_interpolated[(ts_interpolated.country_code == country_code) & (ts_interpolated.sdr_id == sdr_id) & (ts_interpolated.category == 'Cases')]
            days_after = len(df.index) - len(df[df.value > 15].index)
            print days_after
            plot_fit(country_code, sdr_id, train_limit='2014-09-01', days_after=days_after)
            plt.show()
    except KeyError:
        print 'ERROR for', country_code, sdr_id


# In[430]:

def get(country_code, sdr_id, attribute, date=None):
    df = ts_interpolated
    if attribtue not in ts_interpolated.columns: df = covariates
    df = df[(df.country_code == country_code) & (df.sdr_id == sdr_id)]
    # to be finished


# In[ ]:




# In[ ]:




# In[ ]:




# In[ ]:



