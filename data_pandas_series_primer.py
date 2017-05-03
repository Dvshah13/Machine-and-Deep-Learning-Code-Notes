import pandas as pd # importing pandas
import numpy as np # importing numpy for some math functions and to show how pandas and numpy can stack
from pandas import Series, DataFrame # not necessary but will use in script to better explain

## Series - One-dimensional array-like object containing an array of data and associated array of data labels called the index
pd_series = Series([4, 7, -5, 3]) # simplest type of series
print pd_series

# Values in Series
print pd_series.values

# Index in Series
print pd_series.index

# Create Series with your own index
pd_series_cIDX = Series([4, 7, -5, 3], index=['d', 'b', 'a', 'c'])
print pd_series_cIDX

# Get values in certain index
print pd_series_cIDX['a'] # one value
print pd_series_cIDX[['c', 'a', 'd']] # multiple values, nested bracket or raises error

# Numpy array operations (filtering with boolean array, scalar multiplication, applying math function preserves index-value link)
print pd_series_cIDX[pd_series_cIDX > 0] # filtering
print pd_series_cIDX * 2 # scalar multiplication
print np.exp(pd_series_cIDX) # numpy math function

# A Series is essentially a fixed-length, ordered dict, it maps index values to data values.  Thus you can substitute many dict type functions
print 'b' in pd_series_cIDX # returns boolean, in this case true since b index exists in Series
print 'e' in pd_series_cIDX # returns false, not in Series

# Can create Series from Python dict by passing in the dict
ser_data = {'Texas': 55000, 'California': 75000, 'Florida': 45000, 'New York': 63000} # Python Dict
ser_data_pd1 = Series(ser_data) # Convert to pandas Series
print ser_data_pd1

# When only passing a dict, index in the resulting Series will have the dict's keys in sorted order
states = ['Ohio', 'California', 'Florida', 'New York']
ser_data_pd2 = Series(ser_data, index=states)
print ser_data_pd2 # Ohio yields response of NaN (not a number since no value is found in the dict)

# To check for missing data
print pd.isnull(ser_data_pd2) # True for null values
print pd.notnull(ser_data_pd2) # True for non null values
print ser_data_pd2.isnull() # Series built in instance method

# A critical Series feature, aligns differently-indexed data in arithmetic operations
print ser_data_pd1 + ser_data_pd2 # values are aligned by index, since Texas and Ohio appear as NaN for one of the Series, the addition yields NaN as well

# Series Object itself and its index have a name attribute, integrates with other key areas of pandas functionality
ser_data_pd2.name = 'population' # giving overall Series a name
ser_data_pd2.index.name = 'state' # giving index label a name
print ser_data_pd2

# Series index can be altered in place by assignment
pd_series.index = ['Bob', 'Steve', 'Jeff', 'Ryan'] # index now has these labels
print pd_series
