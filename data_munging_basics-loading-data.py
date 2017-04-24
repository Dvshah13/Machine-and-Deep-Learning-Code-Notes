import pandas as pd

## Loading file into pandas DataFrame
# df = pd.read_csv('Data Sets for Code/istanbul_market_data.csv', sep=',', decimal='.')

## Printing head and tail
# print df.head(5)
# print df.tail(5)

## Printing columns
# cols = df.columns.values.tolist()
# print cols

## Get specific column, in a pandas series
# Y = df['Date']
# print Y

## Get multiple columns, in pandas DataFrame
# X = df[['Date', 'FTSE']]
# print X

## Get Shape for for series and DataFrame
# Y = df['Date']
# X = df[['Date', 'FTSE']]
# print X.shape
# print Y.shape

## Dealing with problematic data
## Data set contains unparsed dates, missing values
# bad_data_df = pd.read_csv('Data Sets for Code/istanbul_data_bad.csv', sep=',', decimal='.', parse_dates=[0]) # using parse_dates to convert to date
# print bad_data_df

## Filling in missing values with mean
# impute_vals_df = bad_data_df.fillna(bad_data_df.mean(axis=0))
# print impute_vals_df

## If we have bad lines
# bad_lines_data_df = pd.read_csv('Data Sets for Code/istanbul_data_bad_lines.csv', error_bad_lines = False)
# print bad_lines_data_df

## Using chunks for big data sets
