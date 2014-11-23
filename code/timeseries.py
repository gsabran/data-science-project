
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


## Load and clean the time series data

# Load the data from qdatum.io and the cleaned covariates table

# In[4]:

data = [pd.read_csv('http://api.qdatum.io/v1/pull/' + str(i) +'?format=tsv', sep='\t') for i in range(1, 17)]
covariates = pd.read_csv('../data/merged_covariate_df.csv')


# For each location, the covariate table contains information that might be relevant to Ebola prediction (some data is missing)

# In[5]:

covariates.head()


# Here are the covariate we have access to:

# In[6]:

for c in covariates.columns: print c


# The time serie table contains information on Ebola cases for different location. Relevant information is location, date, cases number, and case status. Most of the categories (cases, deaths...) are cumulatives. New cases category is not

# In[7]:

time_series = data[1].copy()
time_series.head()


# Remove useless values from the time series

# In[8]:

del time_series['pos'] # remove useless columns that prevent duplicates to be identified
del time_series['link']
time_series = time_series[(time_series.value == time_series.value) & (time_series.value != ' ')] # remove NaN values
time_series.value = time_series.value.astype(int) # convert values from string to int
time_series = time_series.drop_duplicates() # remove duplicates


# In[9]:

#Standardize source name
def normalize_source(source):
    source = source.upper()
    if source == 'WHO; GVT': source = 'WHO'
    return source

time_series.sources = time_series.sources.apply(normalize_source)
time_series.sdr_level = time_series.sdr_level.fillna('national')


# Data comes from different sources that might overlap. We will have to select one of them in case of conflict. Some values make no sense and will be removed

# In[10]:

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
plt.title('Ebola cases for Guinea by source')
plot_raw('GN', 0)


##### Note: dates are given as numbers since 12/30/1899 due to an Excel formatting. We have not change it since integer values are good enough

# Select sources and remove decreasing data for cumulative categories. For this we will use a dictionary instead of a dataframe to have more flexibility.

# In[11]:

# transform the dataframe to a list
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
    
# build a dataframe from the clean data
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
print 'Data size before cleaning:', len(time_series.index), 'rows. Data size after cleaning:', len(ts_clean.index), 'rows'
ts_clean.head()


# **We do linear interpolation when the data is missing for some days:**

# In[12]:

# interpolate missing data
first_day = ts_clean.date.min() - 1
interpolated_data = []

# find the day when Ebola is reported for the first time by location
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
                el['type'] = 'interpolate' # keep track of the origin of the data
                el['date'] = i
                el['ebola_already_reported'] = ebola_started # indicate if Ebola has already been reported for this location
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

print 'Data size before interpolation:', len(ts_clean.index),
ts_interpolated = pd.DataFrame(interpolated_data)
print 'rows. Data size after interpolation:', len(ts_interpolated.index), 'rows'
ts_interpolated.head()


# **Dead bodies are an important source of contamination for Ebola. We add a variable that keeps track of the number of recent deaths.**

# In[13]:

# add recent deaths
def add_recent_deaths(interpolated_data, time_series_dict, time_window=5):
    for loc in time_series_dict:
        c = 'Deaths'
        recent_deaths = []
        for d in sorted(time_series_dict[loc][c]):
            el = time_series_dict[loc][c][d].copy()
            removed = 0
            if len(recent_deaths) == time_window: removed = recent_deaths[time_window - 1]
            recent_deaths = recent_deaths[:time_window - 1]
            recent_deaths.append(el['delta'])
            el['category'] = 'Recent Deaths'
            el['value'] = sum(recent_deaths)
            el['delta'] = el['delta'] - removed
            el['type'] = 'interpolate'
            interpolated_data.append(el)
            if 'Recent Deaths' not in time_series_dict[loc]: time_series_dict[loc]['Recent Deaths'] = {}
            time_series_dict[loc]['Recent Deaths'][d] = el
    
add_recent_deaths(interpolated_data, time_series_dict2)

print 'Data size before adding recent deaths:', len(ts_interpolated.index),
ts_interpolated = pd.DataFrame(interpolated_data)
print 'rows. Data size after adding recent deaths:', len(ts_interpolated.index), 'rows'
ts_interpolated[ts_interpolated.category == 'Recent Deaths'].head()


# A helper function to get values from different dataframes:

# In[14]:

# function to get easily values from different dataframes
def get(country_code, sdr_id=0, attribute='Cases', date=None, delta=False):
    try:
        df = covariates
        if attribute in covariates.columns:
            df = df[(df.country_code == country_code) & (df.sdr_id == sdr_id)]
            return df.to_dict()[attribute].popitem()[1]
        else:
            df = ts_interpolated
            if date is not None:
                if type(date) != int and type(date) != np.int64:
                    date = (dt.datetime.strptime(date, '%Y-%m-%d') - dt.datetime(1899, 12, 30)).days
                df = df[df.date == date]
            if attribute in df.columns:
                return df.to_dict()[attribute].popitem()[1] 
            else:
                if delta: k = 'delta'
                else: k = 'value'
                return df.to_dict()[k].popitem()[1]
    except KeyError:
        print 'no value for', country_code, 'sdr id:', sdr_id, attribute,
        if date is not None: print 'date', date
        print
        return None
    
print 'area_sq_km:', get('GN', sdr_id=1, attribute='area_sq_km')
print 'Cases for Guinea on 2014-10-10:', get('GN', sdr_id=0, attribute='Cases', date='2014-10-10')
print 'Ebola already reported in Guinea on 2014-10-10:', get('GN', sdr_id=0, attribute='ebola_already_reported', date='2014-10-10')


## Look at the data

# **Display the time series for a given location:**

# In[15]:

def plot(country_code, sdr_id, categories=ts_interpolated.category.unique(), delta=False):
    df = ts_interpolated[(ts_interpolated.country_code == country_code) & (ts_interpolated.sdr_id == sdr_id)]
    
    for idx, c in enumerate(categories):
        _df = df[(df.category == c)]
        if delta: plt.scatter(_df.date, _df.delta, label='delta ' + c, color=dark2_colors[idx], alpha=0.5)
        else: plt.plot(_df.date, _df.value, label=c)
        
    plt.legend(loc=2)

plot('GN', 0)
plt.title('Ebola evolution in Guinea')
plt.show()


## Model

# We chose use the SIR model (classic epidemiology model) to represent how Ebola spreads:
# 
# $S$ stands for the non infected population, $I$ for the infected persons and $R$ for the removed people (in our case, dead people). This is where the SIR name comes from. $N$ is the population size. It models the disease evolution as follow:
# 
# For every time interval $\Delta T$:
# 
# $\Delta S = - \alpha S \frac{I}{N}$
# 
# $\alpha$ is the rate of contamination, $\frac{I}{N}$ the density of infected persons in the population and $S$ is the scale parameter
# 
# $\Delta I = \alpha S \frac{I}{N} - \beta I$
# 
# $\beta$ is the rate of death of infected persons
# 
# $\Delta R = \beta I$
# 
# This model is well adapted for disease that have an exponential growth (for small $I$, $\frac{S}{N}=\frac{N-I}{N} \sim 1$ and $\Delta I = cst  * I$)
# 
# This give us linear equations in the parameters that we are going to fit with an OLS regression. In the prediction phase, contrary to usual problems, the features $S \frac{I}{N}$ and $I$ are unobserved but can be constructed from the fitted values $\Delta I$ and $\Delta R$. This add variability to the model.

# ***Create the model and target matrixes:***

# In[16]:

def get_first(df, k):
    # return first value in column k
    return df.to_dict()[k].popitem()[1]

def to_X_Y(country_code, sdr_id, train_limit):
    """convert data to OLS matrixes"""
    df = ts_interpolated[(ts_interpolated.country_code == country_code) & (ts_interpolated.sdr_id == sdr_id)]
    X, Y, first_day = [], [], None
    for d in sorted(df.date.unique()):
        if d > train_limit: break
        _df = df[df.date == d]
        infected = _df[_df.category == 'Cases']
        if get_first(infected, 'ebola_already_reported'):
            if first_day is None: first_day = d
            recent_deaths = _df[_df.category == 'Recent Deaths']
            deaths = _df[_df.category == 'Deaths']
            I = get_first(infected, 'value')
            D_recent = get_first(recent_deaths, 'value')
            y = get_first(infected, 'delta')
            x = [I, -I, 0]
            X.append(x)
            Y.append([y])
            
            y = get_first(deaths, 'delta')
            x = [0, I, 0]
            X.append(x)
            Y.append([y])
            
    return np.array(X), np.array(Y), first_day


# ***Fit the model and predict:***
# 
# This is done in three steps:
# 
# - fit the parameters
# 
# - do a prediction of the variations $\Delta I$ and $\Delta D$ based on the fitted paramters and the estimated values of $I$ and $D$
# 
# - reconstruct the cumulatives variables $I$ and $D$

# In[17]:

def fit_data(country_code, sdr_id, train_limit, test_limit, days_after, should_print_table):
    """
    country_code and sdr_id identify the location
    train_limit: date until when we train
    test_limit: date until when we predict
    days_after: allow for some delay after the start of the disease before modeling (the first days usually report for past cases)
    """
    
    # I: Fit the parameters
    
    train_limit = (dt.datetime.strptime(train_limit, '%Y-%m-%d') - dt.datetime(1899, 12, 30)).days
    test_limit = (dt.datetime.strptime(test_limit, '%Y-%m-%d') - dt.datetime(1899, 12, 30)).days
    
    # convert the data to matrix
    X, Y, first_day = to_X_Y(country_code, sdr_id, train_limit)
    if X.shape[0] == 0:
        print 'no data before training limit (', train_limit, ')'
        return None
    
    # remove the beginning of the epidemy
    X_train = X[days_after * 2:, :]
    Y_train = Y[days_after * 2:, :]
    
    if X_train.shape[0] == 0:
        print 'no data before training limit (', train_limit, ')'
        return None
    
    # fit the OLS
    model = sm.OLS(Y_train, X_train)
    results = model.fit()
    
    # print the OLS stats
    if should_print_table: print(results.summary())
    
    # II: Do a prediction
    
    fit = X_train.dot(results.params) # for the training period, we can use the values in X_train
    
    # build the fitted times series step by step after the training period
    # get the last known data
    last_data_known = X_train[-2:, :]
    
    recent_deaths = []
    df = ts_interpolated[(ts_interpolated.country_code == country_code) & (ts_interpolated.sdr_id == sdr_id) & (ts_interpolated.category == 'Deaths')]
    for d in range(train_limit - 10, train_limit):
        _df = df[df.date == d]
        recent_deaths.append(get_first(_df, 'delta'))
    
    # build the predicted data from the estimated rates and the past estimated data
    for i in range(test_limit - train_limit):
        data_interpolated = last_data_known.dot(results.params)
        fit = np.append(fit, data_interpolated)
        delta_I = data_interpolated[0]
        delta_D = data_interpolated[1]
        recent_deaths = recent_deaths[1:]
        recent_deaths.append(delta_D)
        last_data_known = last_data_known + np.array([[delta_I, -delta_I, 0], [0, delta_I, 0]])
        #last_data_known[1, 2] = sum(recent_deaths)
        
    # get the cumulatives values from the variations
    cum_y, cum_death, delta_y, delta_death = [], [], [], []
    y, d_recent = sum(Y[:days_after * 2:2]), sum(Y[1:days_after * 2 + 1:2])
    if type(y) != int: y, d_recent = y[0], d_recent[0]
    for idx, v in enumerate(fit):
        if idx % 2 == 0:
            y += v
            cum_y.append(y)
            delta_y.append(v)
        else:
            d_recent += v
            cum_death.append(d_recent)
            delta_death.append(v)
    return cum_y, cum_death, delta_y, delta_death, range(first_day + days_after, test_limit + 1)


# ***Display the original and fitted values***

# In[18]:

def plot_fit(country_code, sdr_id, train_limit='2014-10-01', test_limit='2014-12-01', days_after=0, should_print_table=False, plot_variations=False):
    fit = fit_data(country_code, sdr_id, train_limit, test_limit, days_after, should_print_table)
    if fit is None: return
    Y_hat, D_hat, delta_Y_hat, delta_D_hat, dates = fit
    
    if not plot_variations:
        plt.plot(dates, Y_hat, label='Cases fit')
        plt.plot(dates, D_hat, label='Deaths fit')
        plot(country_code, sdr_id, ['Cases', 'Deaths'])
        plot_training_limit(train_limit)
        plt.legend(loc=2)
        plt.title('Cases and deaths')
    else:
        plt.plot(dates, delta_Y_hat, label='delta Cases fit')
        plt.plot(dates, delta_D_hat, label='delta Deaths fit')
        plot(country_code, sdr_id, ['Cases', 'Deaths'], delta=True)
        plot_training_limit(train_limit)
        plt.title('variation of cases and deaths')
        plt.legend(loc=2)
    
def plot_training_limit(limit):
    limit = (dt.datetime.strptime(limit, '%Y-%m-%d') - dt.datetime(1899, 12, 30)).days
    ax = plt.axis()
    plt.plot([limit, limit], [0, 1e50], color='r', label='training limit')
    plt.axis(ax)
    
    
country_code, sdr_id = 'GN', 2
plot_fit(country_code, sdr_id, days_after=30)


# The vertical red line mark the end of the training period. A first observation is that the simple SIR model gives a very good fit for a disease that involve complexe mechanisms.
# 
# It is surprising to see the fit much higher than the target (for the number of cases) over almost all the training period. Here is the explanation: the target is not the cumulative number of case, but its variation. In the plot below, we see that the fit of the variation has not such surprising behaviour.

# In[19]:

plot_fit(country_code, sdr_id, days_after=20, plot_variations=True)


# The problem here is that the variations have high variability while the cumulative numbers are quite smooth. For instance a variation of 12 cases over one day fitted at a value of 2 yield a square lost of 100 while four successives variations of 3 fitted at a value of 2 yield a lost of 4. This push the fitted variations up. However, the reality they represent is very close and we don't want such behaviour.
# 
# For this reason, we decided to smooth the variations

# **Smooth variations**

# In[20]:

# smooth the delta
interpolated_data = []
for loc in time_series_dict2:
    for c in time_series_dict2[loc]:
        if c != 'Recent Deaths':
            moving_average = []
            average_length = 15
            cum = 0
            for d in sorted(time_series_dict2[loc][c]):
                el = time_series_dict2[loc][c][d]
                moving_average = moving_average[-average_length + 1:]
                moving_average.append(el['delta'])
                delta = np.array(moving_average).mean()
                el['delta'] = delta
                cum += delta
                el['value'] = cum
                interpolated_data.append(el)
                
add_recent_deaths(interpolated_data, time_series_dict2)

ts_interpolated = pd.DataFrame(interpolated_data)


# We now get a better approximation:

# In[21]:

plot_fit(country_code, sdr_id, days_after=20)


# because the variations are better fitted:

# In[22]:

plot_fit(country_code, sdr_id, days_after=20, plot_variations=True)


## Use the covariates

# Load selected regions that have good data

# In[95]:

selected_regions = pd.read_csv('Selected_SDR.csv', index_col =0)
selected_regions.head()


# Load data for Guinea that needed a special cleaning

# In[96]:

GN = pd.read_csv('Guinea_artificial.csv', index_col =0)
GN.head()


# In[96]:



