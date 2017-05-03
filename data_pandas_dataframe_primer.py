import pandas as pd # importing pandas
import numpy as np # importing numpy for some math functions and to show how pandas and numpy can stack
from pandas import Series, DataFrame # not necessary but will use in script to better explain

## DataFrame - represents tabular, spreadsheet type data structure containing an ordered collection of columns, each of which can be a different value type (numeric, string, boolean, etc).
# DataFrame has both a row and column index; can be thought of as a dict of Series (one for all sharing the same index).
# Compared with something like R's data.frame, row-oriented and column-oriented operations in DataFrame are treated roughly symetrically.  Data is stored as one or more two-dimensional blocks rather than a list, dict or some other collection of One-dimensional arrays.
# Note: Any of the columns returned when indexing a DataFrame is a view on the underlying data not a copy.  Thus, any-in-place modification to the Series will be reflected in the DataFrame.  The column can be explicitly copied using the Series's copy method

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

# Columns can be modified by assignment.  The empty debt category we have could be assigned a scalar value or an array of values
df2['debt'] = 16.5 # scalar value for every row
print df2
df2['debt'] = np.arange(5.) # range of values from 0-4
print df2

# When assigning lists or arrays to a column, the value's length must match the length of the DataFrame.  If you assign a Series, it will be conformed exactly to the DataFrame's index, inserting missing values in an holes
val = Series([-1.2, -1.5, -1.7], index=['two', 'four', 'five'])
df2['debt'] = val
print df2

# Assigning a column that doesn't exist will create a new column.  The del keyword will delete columns as with a dict
df2['eastern'] = df2.state == 'California' # add column eastern and if conditional is met returns True else False
print df2

# Delete column
del df2['eastern']
print df2.columns

# Nested dicts passed into DataFrame
pop = {'Nevada': {2001: 2.4, 2002: 2.9},
        'Ohio': {2000: 1.5, 2001: 1.7, 2002: 3.6}}

# If passed to DataFrame, it will interpret the outer dict keys as the columns and the inner keys as the row indices
df3 = DataFrame(pop)
print df3

# You can always transpose the results (switch axis)
print df3.T

# The keys in the inner dicts are unioned and sorted to form the index in the result.  This isn't true if an explicit index is specified:
print DataFrame(pop, index=[2001, 2002, 2003])

# Dicts of Series are treated much the same way
pdata = {'Ohio': df3['Ohio'][:-1],
        'Nevada': df3['Nevada'][:2]}
df4 = DataFrame(pdata)
print df4

# If DataFrame's index and columns have their name attributes set, there will also be displayed:
df3.index.name = 'year'
df3.columns.name = 'state'
print df3

# Like Series, the values attribute returns the data contained in the DataFrame as a 2D ndarry
print df3.values

# If DataFrame's columns are different dtypes, the dtype of the values array will be chosen to accomodate all of the columns:
print df2.values
