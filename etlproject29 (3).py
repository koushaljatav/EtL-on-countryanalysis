#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np


# In[2]:


df_projects_data=pd.read_csv(r"C:\Users\HP\Downloads\etl\projects_data.csv",dtype="str")


# In[3]:


df_projects_data


# In[4]:


df_population_data=pd.read_csv(r"C:\Users\HP\Downloads\etl\population_data.csv",skiprows=4)


# In[5]:


df_population_data


# In[6]:


df_gdp_data=pd.read_csv(r"C:\Users\HP\Downloads\etl\gdp_data.csv",skiprows=4)


# In[7]:


df_gdp_data


# In[8]:


df_mystery=pd.read_csv(r"C:\Users\HP\Downloads\etl\mystery.csv",encoding="utf16")


# In[9]:


df_mystery


# In[10]:


df_rural_population_percent=pd.read_csv(r"C:\Users\HP\Downloads\etl\rural_population_percent.csv",skiprows=4)


# In[11]:


df_rural_population_percent


# In[12]:


df_electricity_access_percentage=pd.read_csv(r"C:\Users\HP\Downloads\etl\electricity_access_percent.csv",skiprows=4)


# In[13]:


df_electricity_access_percentage


# In[14]:


df_population_data_json=pd.read_json(r"C:\Users\HP\Downloads\etl\population_data.json")


# In[15]:


df_population_data_json


# In[16]:


pip install lxml


# In[17]:


df=pd.read_xml(r"C:\Users\HP\Downloads\etl\population_data.xml")


# In[18]:


df


# In[ ]:





# In[19]:


pip install bs4


# In[20]:


from bs4 import BeautifulSoup

with open(r"C:\Users\HP\Downloads\etl\population_data.xml") as fp:
    soup = BeautifulSoup(fp,"lxml")
i=0
for record in soup.find_all('record'):
    i+=1
    for record in record.find_all('field'):
        print(record['name']+":"+record.text)
    print
    if i==5:
        break


# In[21]:


pip install pymysql


# In[22]:


pip install sqlalchemy


# In[23]:


from sqlalchemy import create_engine
import pandas as pd

username="root"
password="Cybrom#1123"
host="localhost"
port="3306"
database="dummy"

conn=create_engine(f'mysql+pymysql://{username}:{password}@{host}:{port}/{database}')
df_dummy=pd.read_sql("select * from customers",conn)
df_dummy.head()


# In[24]:


import sqlite3


# In[25]:


import pandas as pd


# In[26]:


conn=sqlite3.connect(r"C:\Users\HP\Downloads\etl\population_data.db")
df1=pd.read_sql("select * from population_data",conn)
df1


# In[27]:


df_electricity_access_percentage.drop(['Unnamed: 62'],axis=1,inplace=True)


# In[28]:


df_rural_population_percent.drop(['Unnamed: 62'],axis=1,inplace=True)


# In[29]:


df_rural_population_percent


# In[30]:


df_electricity_access_percentage


# In[ ]:





# In[31]:


df_con=pd.concat([df_electricity_access_percentage,df_rural_population_percent])


# In[32]:


df_con


# In[33]:


df_merge=pd.merge(df_electricity_access_percentage,df_rural_population_percent,on="Country Name",how="outer")
df_merge


# In[34]:


df_electricity_access_percentage.nunique()


# In[35]:


df_electricity_access_percentage["Country Name"].unique()


# In[36]:


df_projects_data.columns


# In[37]:


df_projects_data["countryname"].unique()


# In[38]:


df_projects_data.count()


# In[39]:


df_projects_data.drop(['Unnamed: 56'],axis=1,inplace=True)


# In[40]:


df_projects_data.head()


# Currently, the projects data and the indicators data have different values for country names. My task in this notebook is to clean both data sets so that they have consistent country names. This will allow you to join the two data sets together. Cleaning data, unfortunately, can be tedious and take a lot of your time as a data scientist.
# 
# Why might you want to join these data sets together? What if, for example, you wanted to run linear regression to try to predict project costs based on indicator data? Or you might want to analyze the types of projects that get approved versus the indicator data. For example, do countries with low rates of rural electrification have more rural themed projects?

# In[41]:


df_population_data.drop(['Unnamed: 62'], axis=1, inplace=True)


# In[42]:


df_population_data


# In[43]:


df_projects_data


# The next code cell outputs the unique country names and ISO abbreviations in the population indicator data set.

# In[44]:


df_population_data[['Country Name', 'Country Code']].drop_duplicates()


# In[45]:


df_projects_data['countryname'].unique()


# In[46]:


df_projects_data['Official Country Name'] = df_projects_data['countryname'].str.split(';').str.get(0)
df_projects_data['Official Country Name']


# In[47]:


pip install pycountry


# In[48]:


from pycountry import countries
countries.get(name='India')
countries.lookup('Republic of India')


# The goal is to add the ISO codes to the projects data set. To start, use the pycountry library to make a dictionary mapping the unique countries in 'Official Country Name' to the ISO code.
# 
# Iterate through the unique countries in df_projects. Create a dictionary mapping the 'Country Name' to the alpha_3 ISO abbreviations.
# 
# The dictionary should look like: {'Kingdom of Spain':'ESP'}
# 
# If a country name cannot be found in the pycountry library, add it to a list called country_not_found.

# In[49]:


from collections import defaultdict
country_not_found = []
project_country_abbrev_dict = defaultdict(str) 


for country in df_projects_data['Official Country Name'].drop_duplicates().sort_values():
    try: 
        project_country_abbrev_dict[country] = countries.lookup(country).alpha_3
    except:
        print(country, ' not found')
        country_not_found.append(country)


# In[ ]:





# In[50]:


indicator_countries = df_population_data[['Country Name', 'Country Code']].drop_duplicates().sort_values(by='Country Name')

for country in country_not_found:
    if country in indicator_countries['Country Name'].tolist():
        print(country)


# In[51]:


# now manually create a dictionary that map the country name and country code
country_not_found_mapping = {'Co-operative Republic of Guyana': 'GUY',
             'Commonwealth of Australia':'AUS',
             'Democratic Republic of Sao Tome and Prin':'STP',
             'Democratic Republic of the Congo':'COD',
             'Democratic Socialist Republic of Sri Lan':'LKA',
             'East Asia and Pacific':'EAS',
             'Europe and Central Asia': 'ECS',
             'Islamic  Republic of Afghanistan':'AFG',
             'Latin America':'LCN',
              'Caribbean':'LCN',
             'Macedonia':'MKD',
             'Middle East and North Africa':'MEA',
             'Oriental Republic of Uruguay':'URY',
             'Republic of Congo':'COG',
             "Republic of Cote d'Ivoire":'CIV',
             'Republic of Korea':'KOR',
             'Republic of Niger':'NER',
             'Republic of Kosovo':'XKX','Republic of Rwanda':'RWA',
              'Republic of The Gambia':'GMB',
              'Republic of Togo':'TGO',
              'Republic of the Union of Myanmar':'MMR',
              'Republica Bolivariana de Venezuela':'VEN',
              'Sint Maarten':'SXM',
              "Socialist People's Libyan Arab Jamahiriy":'LBY',
              'Socialist Republic of Vietnam':'VNM',
              'Somali Democratic Republic':'SOM',
              'South Asia':'SAS',
              'St. Kitts and Nevis':'KNA',
              'St. Lucia':'LCA',
              'St. Vincent and the Grenadines':'VCT',
              'State of Eritrea':'ERI',
              'The Independent State of Papua New Guine':'PNG',
              'West Bank and Gaza':'PSE',
              'World':'WLD'}


# In[52]:


project_country_abbrev_dict.update(country_not_found_mapping)


# In[53]:


project_country_abbrev_dict


# In[54]:


df_projects_data['Country Code'] = df_projects_data['Official Country Name'].apply(lambda x: project_country_abbrev_dict[x])


# In[55]:


df_projects_data['Country Code']


# In[56]:


df_projects_data


# In[57]:


df_population_data


# In[58]:


df_population_data.dtypes


# # Calculate the population sum by year for Canada,
# #       the United States, and Mexico.

# In[59]:


keepcol = ['Country Name']
for i in range(1960, 2018, 1):
    keepcol.append(str(i))

df_nafta = df_population_data[(df_population_data['Country Name'] == 'Canada') | (df_population_data['Country Name'] == 'United States') | (df_population_data['Country Name'] == 'Mexico')].iloc[:,]


# In[60]:


df_nafta 


# In[61]:


df_nafta.sum(axis=0)[keepcol]


# In[62]:


df_projects_data.dtypes


# In[63]:


df_projects_data[['totalamt', 'lendprojectcost']].head()


# In[64]:


df_projects_data['totalamt'].sum()


# In[65]:


df_projects_data['totalamt'] = pd.to_numeric(df_projects_data['totalamt'].str.replace(',',""))


# In[66]:


df_projects_data['totalamt']


# float64 int64 bool datetime64 timedelta object
# 
# where timedelta is the difference between two datetimes and object is a string. As you've seen here, you sometimes need to convert data types from one type to another type. Pandas has a few different methods for converting between data types, and here are link to the documentation:
# 
# astype to_datetime to_numeric to_timedelta

# In[67]:


parsed_date = pd.to_datetime('January 1st, 2017')
parsed_date


# In[68]:


parsed_date.month


# In[69]:


parsed_date.year


# In[70]:


parsed_date = pd.to_datetime('5/3/2017 5:30')
parsed_date.month


# In[71]:


parsed_date = pd.to_datetime('3/5/2017 5:30', format='%d/%m/%Y %H:%M')
parsed_date.month


# In[72]:


df_projects_data.head(15)[['boardapprovaldate', 'board_approval_month', 'closingdate']]


# In[73]:


df_projects_data['boardapprovaldate'] = pd.to_datetime(df_projects_data['boardapprovaldate'])
df_projects_data['closingdate'] = pd.to_datetime(df_projects_data['closingdate'])


# In[74]:


df_projects_data['closingdate']


# In[75]:


df_projects_data['boardapprovaldate'].dt.second


# In[76]:


df_projects_data['boardapprovaldate'].dt.month


# In[77]:


df_projects_data['boardapprovaldate'].dt.weekday


# Part 2 - Create new columns
# 
# Now that the boardapprovaldate and closingdates are in datetime formats, create a few new columns in the df_projects data frame:
# 
# approvalyear approvalday approvalweekday closingyear closingday closingweekday

# In[78]:


df_projects_data['approvalyear'] = df_projects_data['boardapprovaldate'].dt.year
df_projects_data['approvalday'] = df_projects_data['boardapprovaldate'].dt.day
df_projects_data['approvalweekday'] = df_projects_data['boardapprovaldate'].dt.weekday
df_projects_data['closingyear'] = df_projects_data['closingdate'].dt.year
df_projects_data['closingday'] = df_projects_data['closingdate'].dt.day
df_projects_data['closingweekday'] = df_projects_data['closingdate'].dt.weekday


# In[79]:


df_projects_data


# #The English alphabet has only 26 letters. #But other languages have many more characters including accents, #tildes and umlauts. As time went on, more encodings were invented to deal with languages other than English. #The utf-8 standard tries to provide a single encoding schema that can encompass all text.
# 
# #The problem is that it's difficult to know what encoding rules were used to make a file unless somebody tells you. #The most common encoding by far is utf-8. Pandas will assume that files are utf-8 when you read them in or write them out.

# In[80]:


from encodings.aliases import aliases

alias_values = set(aliases.values())

for encoding in set(aliases.values()):
        df=pd.read_csv("D:\etl\mystery.csv",encoding="utf-16")
        print('successful', encoding)


# In[81]:


pip install chardet


# In[82]:


import chardet 

with open("D:\etl\mystery.csv", 'rb') as file:
    print(chardet.detect(file.read()))


# In[83]:


df_gdp_data= pd.read_csv('D:\etl\gdp_data.csv', skiprows=4)


# In[84]:


df_gdp_data.drop('Unnamed: 62', axis=1, inplace=True)


# In[85]:


df_gdp_data.isnull().sum()


# In[86]:


df_gdp_data


# In[87]:


import matplotlib.pyplot as plt



df_melt = pd.melt(df_gdp_data, id_vars=['Country Name', 'Country Code', 'Indicator Name', 'Indicator Code'], var_name='year', value_name='GDP')


df_melt['year'] = pd.to_datetime(df_melt['year'])

def plot_results(column_name):
    fig, ax = plt.subplots(figsize=(8,6))

    df_melt[(df_melt['Country Name'] == 'Afghanistan') | 
            (df_melt['Country Name'] == 'Albania') | 
            (df_melt['Country Name'] == 'Honduras')].groupby('Country Name').plot('year', column_name, legend=True, ax=ax)
    ax.legend(labels=['Afghanistan', 'Albania', 'Honduras'])
    
plot_results('GDP')


# Afghanistan and Albania are missing data, which show up as gaps in the results.
# 
# Exercise - Part 1
# 
# Your first task is to calculate mean GDP for each country and fill in missing values with the country mean. This is a bit tricky to do in pandas

# In[88]:


df_melt['GDP_filled'] = df_melt.groupby('Country Name')['GDP'].transform(lambda x: x.fillna(x.mean()))


# In[89]:


plot_results('GDP_filled')


# This is somewhat of an improvement. At least there is no missing data; however, because GDP tends to increase over time, the mean GDP is probably not the best way to fill in missing values for this particular case. Next, try using forward fill to deal with any missing values.
# 
# Exercise Part -2 The pandas fillna method has a forward fill option. For example, if you wanted to use forward fill on the GDP dataset, you could execute df_melt[GDP].fillna(method='ffill'). However, there are two issues with that code.
# 
# You want to first make sure the data is sorted by year You need to group the data by country name so that the forward fill stays within each country

# In[90]:


df_melt['GDP_ffill'] = df_melt.sort_values('year').groupby('Country Name')['GDP'].fillna(method='ffill')


# In[91]:


plot_results('GDP_ffill')


# In[92]:


df_melt['GDP_bfill'] = df_melt.sort_values('year').groupby('Country Name')['GDP'].fillna(method='bfill')


# In[93]:


plot_results('GDP_bfill')


# In[94]:


df_melt['GDP_ff_bf'] = df_melt.sort_values('year').groupby('Country Name')['GDP'].fillna(method='ffill').fillna(method='bfill')


df_melt


# Duplicate Data
# 
# A data set might have duplicate data: in other words, the same record is represented multiple times. Sometimes, it's easy to find and eliminate duplicate data like when two records are exactly the same duplicate data is hard to spot.
# 
# From the World Bank GDP data, count the number of countries that have had a project totalamt greater than 1 billion dollars (1,000,000,000). To get the count, you'll have to remove duplicate data rows.

# In[95]:


df_projects_data['totalamt'] = df_projects_data['totalamt'].astype(str).str.replace(',', '')


df_projects_data['totalamt'] = pd.to_numeric(df_projects_data['totalamt'], errors='coerce')

df_projects_data['countryname'] = df_projects_data['countryname'].str.split(';', expand=True)[0]
df_projects_data['boardapprovaldate'] = pd.to_datetime(df_projects_data['boardapprovaldate'])


# In[100]:


df_projects_data[df_projects_data['totalamt'] > 1000000000]['countryname'].nunique()


# In[101]:


df_projects_data['totalamt'] 


# Dummy Variables In this exercise, I'll create dummy variables from the projects data set. The idea is to transform categorical data like this:
# 
# Project ID Project Category 0 Energy 1 Transportation 2 Health 3 Employment
# 
# The reasoning behind these transformations is that machine learning algorithms read in numbers not text. Text needs to be converted into numbers. You could assign a number to each category like 1, 2, 3, and 4. But a categorical variable has no inherent order, so you want to reflect this in your features.
# 
# Pandas makes it very easy to create dummy variables with the get_dummies method. In this exercise, you'll create dummy variables from the World Bank projects data; however, there's a caveat. The World Bank data is not particularly clean, so you'll need to explore and wrangle the data first.

# In[102]:


sector = df_projects_data.copy()
sector = sector[['project_name', 'lendinginstr', 'sector1', 'sector2', 'sector3', 'sector4', 'sector5', 'sector',
          'mjsector1', 'mjsector2', 'mjsector3', 'mjsector4', 'mjsector5',
          'mjsector', 'theme1', 'theme2', 'theme3', 'theme4', 'theme5', 'theme ',
          'goal', 'financier', 'mjtheme1name', 'mjtheme2name', 'mjtheme3name',
          'mjtheme4name', 'mjtheme5name']]


# In[103]:


sector


# In[104]:


100 * sector.isnull().sum() / sector.shape[0]


# In[105]:


uniquesectors1 = sector['sector1'].sort_values().unique()
uniquesectors1


# In[106]:


print('Number of unique values in sector1:', len(uniquesectors1))


# In[107]:


import numpy as np

sector['sector1'] = sector['sector1'].replace('!$!0', np.nan)

sector['sector1'] = sector['sector1'].replace('!.+', '', regex=True)


sector['sector1'] = sector['sector1'].replace('^(\(Historic\))', '', regex=True)

print('Number of unique sectors after cleaning:', len(list(sector['sector1'].unique())))
print('Percentage of null values after cleaning:', 100 * sector['sector1'].isnull().sum() / sector['sector1'].shape[0]) 


# In[108]:


sector['sector1'].unique()


# In[109]:


dummies = pd.DataFrame(pd.get_dummies(sector['sector1']))

df_projects_data['year'] = df_projects_data['boardapprovaldate'].dt.year
df = df_projects_data[['totalamt','year']]       
df_final = pd.concat([df, dummies], axis=1)

df_final


# In[110]:


dummies


# In[111]:


gdp = pd.read_csv("D:\etl\gdp_data.csv", skiprows=4)
gdp.drop(['Unnamed: 62', 'Country Code', 'Indicator Name', 'Indicator Code'], inplace=True, axis=1)
population = pd.read_csv("D:\etl\population_data.csv", skiprows=4)
population.drop(['Unnamed: 62', 'Country Code', 'Indicator Name', 'Indicator Code'], inplace=True, axis=1)


# In[112]:


gdp_melt = gdp.melt(id_vars=['Country Name'], 
                    var_name='year', 
                    value_name='gdp')


gdp_melt['gdp'] = gdp_melt.sort_values('year').groupby('Country Name')['gdp'].fillna(method='ffill').fillna(method='bfill')

population_melt = population.melt(id_vars=['Country Name'], 
                                  var_name='year', 
                                  value_name='population')


population_melt['population']=population_melt.sort_values('year').groupby('Country Name')['population'].fillna(method='ffill').fillna(method='bfill')


df_country = gdp_melt.merge(population_melt,on=('Country Name','year'))

df_2016=df_country[df_country['year']=='2016']


# In[113]:


df_2016.head(10)


# In[114]:


import matplotlib.pyplot as plt

df_2016.plot('population',kind='box')
df_2016.plot('gdp',kind='box')
plt.show()


# In[115]:


population_2016=df_2016[['Country Name','population']]

Q1=population_2016['population'].quantile(0.25)


Q3=population_2016['population'].quantile(0.75)

IQR=Q3 - Q1


max_value = Q3 + 1.5 * IQR
min_value = Q1 - 1.5 * IQR


population_outliers = population_2016[(population_2016['population'] > max_value) | (population_2016['population'] < min_value)]
population_outliers


# In[116]:


non_countries = ['World',
 'High income',
 'OECD members',
 'Post-demographic dividend',
 'IDA & IBRD total',
 'Low & middle income',
 'Middle income',
 'IBRD only',
 'East Asia & Pacific',
 'Europe & Central Asia',
 'North America',
 'Upper middle income',
 'Late-demographic dividend',
 'European Union',
 'East Asia & Pacific (excluding high income)',
 'East Asia & Pacific (IDA & IBRD countries)',
 'Euro area',
 'Early-demographic dividend',
 'Lower middle income',
 'Latin America & Caribbean',
 'Latin America & the Caribbean (IDA & IBRD countries)',
 'Latin America & Caribbean (excluding high income)',
 'Europe & Central Asia (IDA & IBRD countries)',
 'Middle East & North Africa',
 'Europe & Central Asia (excluding high income)',
 'South Asia (IDA & IBRD)',
 'South Asia',
 'Arab World',
 'IDA total',
 'Sub-Saharan Africa',
 'Sub-Saharan Africa (IDA & IBRD countries)',
 'Sub-Saharan Africa (excluding high income)',
 'Middle East & North Africa (excluding high income)',
 'Middle East & North Africa (IDA & IBRD countries)',
 'Central Europe and the Baltics',
 'Pre-demographic dividend',
 'IDA only',
 'Least developed countries: UN classification',
 'IDA blend',
 'Fragile and conflict affected situations',
 'Heavily indebted poor countries (HIPC)',
 'Low income',
 'Small states',
 'Other small states',
 'Caribbean small states',
 'Pacific island small states']

df_2016 = df_2016[~df_2016['Country Name'].isin(non_countries)]



# In[117]:


population_2016=df_2016[['Country Name','population']]
 
Q1=population_2016['population'].quantile(0.25)

Q3=population_2016['population'].quantile(0.75)

IQR=Q3 - Q1


max_value=Q3 + 1.5 * IQR
min_value=Q1 - 1.5 * IQR


population_outliers = population_2016[(population_2016['population'] > max_value) | (population_2016['population'] < min_value)]
population_outliers


# In[118]:


gdp_2016=df_2016[['Country Name','gdp']]

Q1=gdp_2016['gdp'].quantile(0.25)

Q3=gdp_2016['gdp'].quantile(0.75)


IQR=Q3 - Q1

max_value=Q3 + 1.5 * IQR
min_value=Q1 - 1.5 * IQR

gdp_outliers=gdp_2016[(gdp_2016['gdp'] > max_value) | (gdp_2016['gdp'] < min_value)]
gdp_outliers


# In[119]:


list(set(population_outliers['Country Name']).intersection(gdp_outliers['Country Name']))


# In[120]:


list(set(population_outliers['Country Name']) - set(gdp_outliers['Country Name']))


# In[121]:


list(set(gdp_outliers['Country Name']) - set(population_outliers['Country Name']))


# In[122]:


x = list(df_2016['population'])
y = list(df_2016['gdp'])
text = df_2016['Country Name']

fig, ax = plt.subplots(figsize=(15,10))
ax.scatter(x, y)
plt.title('GDP vs Population')
plt.xlabel('population')
plt.ylabel('GDP')
for i, txt in enumerate(text):
    ax.annotate(txt, (x[i],y[i]))


# In[123]:


df_no_large = (df_2016['Country Name'] != 'United States') & (df_2016['Country Name'] != 'India') & (df_2016['Country Name'] != 'China')
x = list(df_2016[df_no_large]['population'])
y = list(df_2016[df_no_large]['gdp'])
text = df_2016[df_no_large]['Country Name']

fig, ax = plt.subplots(figsize=(15,10))
ax.scatter(x, y)
plt.title('GDP vs Population')
plt.xlabel('population')
plt.ylabel('GDP')
for i, txt in enumerate(text):
    ax.annotate(txt, (x[i],y[i]))


# In[124]:


from sklearn.linear_model import LinearRegression


model = LinearRegression()
model.fit(df_2016['population'].values.reshape(-1, 1), df_2016['gdp'].values.reshape(-1, 1))

inputs = np.linspace(1, 2000000000, num=50)
predictions = model.predict(inputs.reshape(-1,1))

df_2016.plot('population', 'gdp', kind='scatter')
plt.plot(inputs, predictions)


# In[125]:


df_2016[df_2016['Country Name'] != 'United States'].plot('population', 'gdp', kind='scatter')
# plt.plot(inputs, predictions)
model.fit(df_2016[df_2016['Country Name'] != 'United States']['population'].values.reshape(-1, 1), 
          df_2016[df_2016['Country Name'] != 'United States']['gdp'].values.reshape(-1, 1))
inputs = np.linspace(1, 2000000000, num=50)
predictions = model.predict(inputs.reshape(-1,1))
plt.plot(inputs, predictions)


# Notice that the code now ouputs a GDP value of 5.26e+12 when population equals 1e9. In other words, removing the United States shifted the linear regression line down.
# 
# Data scientists sometimes have the task of creating an outlier removal model. In this exercise, you've used the Tukey rule. There are other one-dimensional models like eliminating data that is far from the mean. There are also more sophisticated models that take into account multi-dimensional data.
# 
# Eliminating Outliers
# 
# Eliminating outliers is a big topic. There are many different ways to eliminate outliers. A data engineer's job isn't necessarily to decide what counts as an outlier and what does not. A data scientist would determine that. The data engineer would code the algorithms that eliminate outliers from a data set based on any criteria that a data scientist has decided.

# In[126]:


def tukey_rule(data_frame, column_name):
    data = data_frame[column_name]
    Q1 = data.quantile(0.25)
    Q3 = data.quantile(0.75)

    IQR = Q3 - Q1

    max_value = Q3 + 1.5 * IQR
    min_value = Q1 - 1.5 * IQR
    
    return data_frame[(data_frame[column_name] < max_value) & (data_frame[column_name] > min_value)]


# In[127]:


df_outlier_removed = df_2016.copy()

for column in ['population','gdp']:
    df_outlier_removed = tukey_rule(df_outlier_removed, column)


# In[128]:


x = list(df_outlier_removed['population'])
y = list(df_outlier_removed['gdp'])
text = df_outlier_removed['Country Name']

fig, ax = plt.subplots(figsize=(15,10))
ax.scatter(x, y)
plt.title('GDP vs Population')
plt.xlabel('GDP')
plt.ylabel('Population')
for i, txt in enumerate(text):
    ax.annotate(txt, (x[i],y[i]))


# In[129]:


def x_min_max(data):
    minimum = min(data)
    maximum = max(data)
    return minimum, maximum

x_min_max(df_2016['gdp'])


# In[130]:


def normalize(x, x_min, x_max):
    # Complete this function
    # The input is a single value 
    # The output is the normalized value
    return (x - x_min) / (x_max - x_min)


# In[131]:


class Normalizer():
    def __init__(self, dataframe):
        self.params = []

        for column in dataframe.columns:
            self.params.append(x_min_max(dataframe[column]))
    def x_min_max(data):
        minimum = min(data)
        maximum = max(data)
        return minimum, maximum
        
    def normalize_data(self, x):
        normalized = []
        for i, value in enumerate(x):
            x_max = self.params[i][1]
            x_min = self.params[i][0]
            normalized.append((x[i] - x_min) / (x_max - x_min))
        return normalized
        


# In[132]:


gdp_normalizer = Normalizer(df_2016[['gdp', 'population']])


# In[134]:


gdp_normalizer.params


# In[135]:


gdp_normalizer.normalize_data([13424475000000.0, 1300000000])


# In[136]:


df_2016 = df_2016[~df_2016['Country Name'].isin(non_countries)]
df_2016.reset_index(inplace=True, drop=True)


# In[137]:


df_2016['gdppercapita'] = df_2016['gdp'] / df_2016['population']


# In[139]:


df_2016


# In[188]:


gdp = pd.read_csv("D:\etl\gdp_data.csv", skiprows=4)
gdp.drop(['Unnamed: 62', 'Indicator Name', 'Indicator Code'], inplace=True, axis=1)
population = pd.read_csv("D:\etl\population_data.csv", skiprows=4)
population.drop(['Unnamed: 62', 'Indicator Name', 'Indicator Code'], inplace=True, axis=1)



gdp_melt = gdp.melt(id_vars=['Country Name', 'Country Code'], 
                    var_name='year', 
                    value_name='gdp')

gdp_melt['gdp'] = gdp_melt.sort_values('year').groupby(['Country Name', 'Country Code'])['gdp'].fillna(method='ffill').fillna(method='bfill')

population_melt = population.melt(id_vars=['Country Name', 'Country Code'], 
                                  var_name='year', 
                                  value_name='population')


population_melt['population'] = population_melt.sort_values('year').groupby('Country Name')['population'].fillna(method='ffill').fillna(method='bfill')

df_indicator = gdp_melt.merge(population_melt, on=('Country Name', 'Country Code', 'year'))
non_countries = ['World',
 'High income',
 'OECD members',
 'Post-demographic dividend',
 'IDA & IBRD total',
 'Low & middle income',
 'Middle income',
 'IBRD only',
 'East Asia & Pacific',
 'Europe & Central Asia',
 'North America',
 'Upper middle income',
 'Late-demographic dividend',
 'European Union',
 'East Asia & Pacific (excluding high income)',
 'East Asia & Pacific (IDA & IBRD countries)',
 'Euro area',
 'Early-demographic dividend',
 'Lower middle income',
 'Latin America & Caribbean',
 'Latin America & the Caribbean (IDA & IBRD countries)',
 'Latin America & Caribbean (excluding high income)',
 'Europe & Central Asia (IDA & IBRD countries)',
 'Middle East & North Africa',
 'Europe & Central Asia (excluding high income)',
 'South Asia (IDA & IBRD)',
 'South Asia',
 'Arab World',
 'IDA total',
 'Sub-Saharan Africa',
 'Sub-Saharan Africa (IDA & IBRD countries)',
 'Sub-Saharan Africa (excluding high income)',
 'Middle East & North Africa (excluding high income)',
 'Middle East & North Africa (IDA & IBRD countries)',
 'Central Europe and the Baltics',
 'Pre-demographic dividend',
 'IDA only',
 'Least developed countries: UN classification',
 'IDA blend',
 'Fragile and conflict affected situations',
 'Heavily indebted poor countries (HIPC)',
 'Low income',
 'Small states',
 'Other small states',
 'Not classified',
 'Caribbean small states',
 'Pacific island small states']

df_indicator  = df_indicator[~df_indicator['Country Name'].isin(non_countries)]
df_indicator.reset_index(inplace=True, drop=True)

df_indicator.columns = ['countryname', 'countrycode', 'year', 'gdp', 'population']

df_indicator.head()


# In[189]:


df_projects_data['countryname'] = df_projects_data['countryname'].str.split(';').str.get(0)


from collections import defaultdict
country_not_found = [] 
project_country_abbrev_dict = defaultdict(str) 


for country in df_projects_data['countryname'].drop_duplicates().sort_values():
    try: 
        project_country_abbrev_dict[country] = countries.lookup(country).alpha_3
    except:
        country_not_found.append(country)
        


# In[190]:


country_not_found_mapping = {'Co-operative Republic of Guyana': 'GUY',
             'Commonwealth of Australia':'AUS',
             'Democratic Republic of Sao Tome and Prin':'STP',
             'Democratic Republic of the Congo':'COD',
             'Democratic Socialist Republic of Sri Lan':'LKA',
             'East Asia and Pacific':'EAS',
             'Europe and Central Asia': 'ECS',
             'Islamic  Republic of Afghanistan':'AFG',
             'Latin America':'LCN',
              'Caribbean':'LCN',
             'Macedonia':'MKD',
             'Middle East and North Africa':'MEA',
             'Oriental Republic of Uruguay':'URY',
             'Republic of Congo':'COG',
             "Republic of Cote d'Ivoire":'CIV',
             'Republic of Korea':'KOR',
             'Republic of Niger':'NER',
             'Republic of Kosovo':'XKX',
             'Republic of Rwanda':'RWA',
              'Republic of The Gambia':'GMB',
            'Republic of Togo':'TGO',
              'Republic of the Union of Myanmar':'MMR',
              'Republica Bolivariana de Venezuela':'VEN',
              'Sint Maarten':'SXM',
              "Socialist People's Libyan Arab Jamahiriy":'LBY',
              'Socialist Republic of Vietnam':'VNM',
              'Somali Democratic Republic':'SOM',
              'South Asia':'SAS',
              'St. Kitts and Nevis':'KNA',
              'St. Lucia':'LCA',
              'St. Vincent and the Grenadines':'VCT',
              'State of Eritrea':'ERI',
              'The Independent State of Papua New Guine':'PNG',
              'West Bank and Gaza':'PSE',
              'World':'WLD'}

df_projects_data.columns = df_projects_data.columns.str.lower()

df_projects_data['countrycode'] = df_projects_data['countryname'].apply(
    lambda x: project_country_abbrev_dict.get(x, 'Unknown')
)


df_projects_data['totalamt'] = pd.to_numeric(df_projects_data['totalamt'].astype(str).str.replace(',', ''), errors='coerce')


df_projects_data = df_projects_data[['id', 'countryname', 'countrycode', 'totalamt', 'year']]


df_projects_data.head()


# In[200]:


df_merged = df_projects_data.merge(df_indicator, how='left', on=['countrycode', 'year'])


# In[201]:


df_merged[(df_merged['year'] == '2017') & (df_merged['countryname_y'] == 'Jordan')]


# In[202]:


df_merged.to_json('countrydata.json', orient='records')


# In[203]:


df_merged.to_csv('countrydata.csv', index=False)


# In[204]:


host = "localhost"  
user = "root"  
password = "Cybrom#1123" 
database = "worldbank"  


conn = pymysql.connect(host=host, user=user, password=password, database=database)
engine = create_engine(f"mysql+pymysql://{user}:{password}@{host}/{database}")

df_merged.to_sql('merged', con=engine, if_exists='replace', index=False)


query = 'SELECT * FROM merged WHERE year = "2017" AND countrycode = "BRA"'
result = pd.read_sql(query, con=engine)


print(result.head())


conn.close()


# In[206]:


host = "localhost"  
user = "root" 
password = "Cybrom#1123"  
database = "worldbank" 


conn = pymysql.connect(host=host, user=user, password=password, database=database)
engine = create_engine(f"mysql+pymysql://{user}:{password}@{host}/{database}")

df_indicator.to_sql('indicator', con=engine, if_exists='replace', index=False)
df_projects_data.to_sql('projects', con=engine, if_exists='replace', index=False)

query = '''
SELECT * 
FROM projects 
LEFT JOIN indicator 
ON projects.countrycode = indicator.countrycode 
   AND projects.year = indicator.year 
WHERE projects.year = "2017" 
  AND projects.countrycode = "BRA"
'''
result = pd.read_sql(query, con=engine)


print(result.head())

conn.close()


# In[ ]:





# In[ ]:




