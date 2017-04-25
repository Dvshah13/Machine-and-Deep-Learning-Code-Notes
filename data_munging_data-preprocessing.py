import pandas as pd

## Loading file into pandas DataFrame
df = pd.read_csv('Data Sets for Code/istanbul_market_data.csv', sep=',', decimal='.')

## creating a mask function to a Boolean value to a condition I make (here true or false for TL Based ISE greater then 0.02)
# mask_feature = df['TL Based ISE'] > 0.02
print mask_feature

## use a selection mask to substitute label with New Label for practice, change date to New Label
mask_target = df['Date'] == '5-Jan-09'
df.loc[mask_target, 'Date'] = 'New Label'

## to see new list of labels in the column, categorical, use unique() method, handy method to initially evaluate the dataset
print df['Date'].unique()

## if you want to see some statistics about each feature, you can group column accordingly
grouped_targets_mean = df.groupby(['TL Based ISE']).mean() # group by mean
print grouped_targets_mean

# grouped_targets_var = df.groupby(['TL Based ISE']).var() # group by variance
print grouped_targets_var

## if you need to sort observations using a function, use sort() method
print df.sort_index(by='FTSE').head()

# if your dataset contains a time series and you need to apply a rolling operation to it (in the case of noisy data points), you can do the following
smooth_time_series = pd.rolling_mean(time_series, 5) # can do the same for median
print smooth_time_series

## Dealing with an index column in a csv file
df = pd.read_csv('filename.csv', index_col=0)

## extract value of column and row
print df['Date'][90] # returns value from date at row 90, specify column then row
# can also use .loc() method
print df.loc[90, 'Date'] # returns value from date at row 90, specify row then column
# can also use .ix() method
print df.ix[90, 'Date']
# using .ix() method with column number
print df.ix[90, 0]
# using a full-optimized function that specifies positions, iloc()
print df.iloc[2,2] # specify the cell using row and column number
# specifying the lists of indexes instead of scalars
print df[['Date', 'FTSE']][0:2]
