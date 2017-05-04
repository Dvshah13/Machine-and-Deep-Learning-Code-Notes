import pandas as pd # importing pandas
import numpy as np # importing numpy for some math functions and to show how pandas and numpy can stack
from pandas import Series, DataFrame # not necessary but will use in script to better explain

## Pandas Index Objects - responsible for holding the axis label and other metadata (like axis name or names).  Any array or other sequence of labels used when constructing a Series or DataFrame is internally converted to an Index:

# Pandas Object
obj = Series(range(3), index=['a', 'b', 'c'])
index = obj.index
print index
print index[1:]

# Index objects are immutable and thus can't be modified by the user.  Immutability is important so that Index objects can be safely shared among data structures
index = pd.Index(np.arange(3))
obj2 = Series([1.5, -2.5, 0], index=index)
print obj2.index is index # boolean value to check truthiness

# In addition to being array-like, an Index also functions as a fixed-size set and can use functions to check for truthiness on columns and index for a DataFrame
data = {'state': ['California', 'California', 'California', 'Texas', 'Texas'],
        'year': ['2010', '2011', '2012', '2010', '2011'],
        'pop': [30.7, 32.6, 33.7, 25.6, 26.7]}
df = DataFrame(data)
print 'California' in df.columns # returns True
print 2003 in df.index # returns False
# Ton of methods and properties you can use on an index including: append, diff, intersection, union, isin, delete, drop, insert, is_montonic, is_unique, unique

# Reindexing - Critical method on Pandas which means to create a new object with the data conformed to a new index.
obj = Series([4.5, 7.2, -5.3, 3.6], index=['d', 'b', 'a', 'c'])
print obj
# Calling reindex on this Series rearranges the data according to the new index, introducing missing values if any index values were not already present:
obj2 = obj.reindex(['a', 'b', 'c', 'd', 'e'])
print obj2

# Fill in missing values if they exists in Series
obj2_fill = obj.reindex(['a', 'b', 'c', 'd', 'e'], fill_value=0)
print obj2_fill

# For ordered data like time series, it may be desirable to do some interpolation or filling of values when reindexing.  The method option allows us to do this, using a method such as ffill which forward fills the values:
obj3 = Series(['blue', 'purple', 'yellow'], index=[0, 2, 4])
obj3_ffill = obj3.reindex(range(6), method='ffill') # can also use other method options such as pad (ffill), bfill/backfill (fill values backward)
print obj3_ffill

# With DataFrame, reindex can alter either the (row) index, columns or both.
# When passed just a sequence, the rows are reindexed in the result:
frame = DataFrame(np.arange(9).reshape((3, 3)), index=['a', 'c', 'd'], columns=['Ohio', 'Texas', 'California'])
print frame
frame2 = frame.reindex(['a', 'b', 'c', 'd'])
print frame2
# The columns can be reindexed using the columns keyword:
states = ['Texas', 'Utah', 'California']
print frame.reindex(columns=states)
# Both can be reindexed in one shot, though interpolation will only apply row-wise (axis 0):
print frame.reindex(index=['a', 'b', 'c', 'd'], method='ffill', columns=states)
# Reindexing can be done more succinctly by label-indexing with ix:
print frame.ix[['a', 'b', 'c', 'd'], states]
# Some reindex function arguments include: index, method, fill_value, limit, level, copy

# Dropping entries from an axis - dropping one or more entries from an axis is easy if you have an index array or list without those entries. As that can require a bit of munging and set logic, the drop method will return a new object with the indicated value or values deleted from an axis:
obj = Series(np.arange(5.), index=['a', 'b', 'c', 'd', 'e'])
new_obj = obj.drop('c') # drop c from Series
print new_obj
new_obj = obj.drop(['d', 'c'])
print new_obj
# With DataFrame, index values can be deleted from either axis:
data = DataFrame(np.arange(16).reshape((4, 4)), index=['Ohio', 'Colorado', 'Utah', 'New York'], columns=['one', 'two', 'three', 'four'])
new_data = data.drop(['Colorado', 'Ohio'])
print new_data
new_data = data.drop('two', axis=1)
print new_data
new_data = data.drop(['two', 'four'], axis=1)
print new_data

# Indexing, Selection, Filtering
# Series indexing works analogously to NumPy array indexing, except you can use the Series's index value instead of only integers
obj = Series(np.arange(4.), index=['a', 'b', 'c', 'd'])
print obj['b'] # returns 1.0
print obj[1] # also returns 1.0
print obj[2:4] # returns c: 2, d: 3
print obj[['b', 'a', 'd']] # returns b: 1, a: 0, d: 3
print obj[[1,3]] # returns b: 1, d: 3
print obj[obj < 2] # returns a: 0, b: 1
# Slicing with labels behaves differently than normal Python slicing in that the endpoint is inclusive:
print obj['b':'c'] # returns b: 1, c: 2
# Setting using these methods works just as you would expect:
obj['b': 'c'] = 5
print obj
# Indexing into a DataFrame is for retrieving one or more columns either with a single value or sequence:
data = DataFrame(np.arange(16).reshape((4, 4)), index=['Ohio', 'Colorado', 'Utah', 'New York'], columns=['one', 'two', 'three', 'four'])
print data
print data['two'] # returns column two
print data['three', 'one'] # returns columns three and one
# Indexing like this has a few special cases.  First, selecting rows by slicing or a boolean 
