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

## Using chunks for big data sets, this example prints the chunk shape and each chunk
# df = pd.read_csv('Data Sets for Code/istanbul_market_data.csv', sep=',', decimal='.', chunksize=10)
# for chunk in df:
#     print chunk.shape
#     print chunk

## Using iterator to load big data set
# df = pd.read_csv('Data Sets for Code/istanbul_market_data.csv', sep=',', decimal='.', iterator = True)
# print df.get_chunk(10).shape
# print df.get_chunk(2)

## Using csv packages, DictReader, to iterate small chunks of data from files
# import csv
# with open('Data Sets for Code/istanbul_market_data.csv', 'rb') as data_stream:
#     for n, row in enumerate(csv.DictReader(data_stream, dialect="excel")):
#         if n == 0:
#             print n, row
#         else:
#             break

## Using csv packages, reader, to iterate small chunks of data from files
import csv
# with open('Data Sets for Code/istanbul_market_data.csv', 'rb') as data_stream:
#     for n, row in enumerate(csv.reader(data_stream, dialect="excel")):
#         if n == 0:
#             print row
#         else:
#             break

## Using batch parameter to load data
# filename = 'Data Sets for Code/istanbul_market_data.csv'
# def batch_read(filename, batch=5):
#     # open data stream
#     with open(filename, 'rb') as data_stream:
#         # reset the batch
#         batch_output = list()
#         #iterate over filename
#         for n, row in enumerate(csv.reader(data_stream, dialect='excel')):
#             #if the batch is of the right size
#             if n > 0 and n % batch == 0:
#                 # yield back the batch as an ndarray
#                 yield(np.array(batch_output))
#                 # reset the batch and restart
#                 batch_output = list()
#             # otherwise add the row to the batch
#             batch_output.append(row)
#         # when the loop is over, yield what's left
#         yield(np.array(batch_output))
# import numpy as np
# for batch_input in batch_read(filename, batch=5):
#     print batch_input
#     break
