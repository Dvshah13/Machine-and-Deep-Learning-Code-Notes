import pandas as pd # importing pandas
import numpy as np # importing numpy for some math functions and to show how pandas and numpy can stack
from pandas import Series, DataFrame # not necessary but will use in script to better explain

## DataFrame - represents tabular, spreadsheet type data structure containing an ordered collection of columns, each of which can be a different value type (numeric, string, boolean, etc).
# DataFrame has both a row and column index; can be thought of as a dict of Series (one for all sharing the same index).
# Compared with something like R's data.frame, row-oriented and column-oriented operations in DataFrame are treated roughly symetrically.  Data is stored as one or more two-dimensional blocks rather than a list, dict or some other collection of One-dimensional arrays.

# Constructing a DataFrame using a dict of equal-length lists or NumPy arrays
data = {'state': ['California', 'California', 'California', 'Texas', 'Texas'],
        'year': ['2010', '2011', '2012', '2010', '2011'],
        'pop': [30.7, 32.6, 33.7, 25.6, 26.7]}
df = DataFrame(data)
print df

# If you specify a sequence of columns, the DataFrame's columns will be exactly what you pass:
df_cols = DataFrame(data, columns=['year', 'state', 'pop']) # if column name does match dict you will get a new column of NaN values
print df_cols

# As with Series, if you pass a column not in the data, you get NaN values
df2 = DataFrame(data, columns=['year', 'state', 'pop', 'debt'],
                index=['one', 'two', 'three', 'four', 'five'])
print df2
print df2.columns # prints the columns

# A column in a DataFrame can be retrieved as a Series wither by dict-like notation or by attribute
print df2['state'] # This Series has the same index as the DataFrame and name attribue has been appropriately set
print df2['year'] # This Series has the same index as the DataFrame and name attribue has been appropriately set

# Rows can be retrieved by a couple of methods: here is the ix indexing field:
print df2.ix['three']
