## Considered one of the first steps in the data science process, it's a process required to better understand the dataset, check its features and shape, validate some hypothesis you have in mind and have a preliminary idea about the next step that you want to persue in the following data science tasks.

## Loading dataset
import pandas as pd

## Loading file into pandas DataFrame
df = pd.read_csv('Data Sets for Code/istanbul_market_data.csv', sep=',', decimal='.')

## Printing head
print df.head(5)

## Now that dataset is loaded, you can get some insight by using the .describe() method, we get the count, mean, standard deviation, min and max values, some percentiles (25, 50 and 75 percent) which gives you a good idea about the distribution of each feature
print df.describe()

## You can also visualize the information by using the box plot method
df.boxplot()

## If you need other quantile values, you can use the .quantile() method
print df.quantile([0.1,0.9]) # the 10 percent and 90 percent values

## To calculate the median, you can use the .median method, similarly you can use .mean() and .std() methods for mean and standard deviation respectively.  In the case of categorical features to get information about the levels (that is, the different values the feature assumes), you can use the .unique() method
df.median() # median
df.feature_name.unique() # type in the feature name you want to examine and you can get information about the levels

## To examine the relation between features, you can create a co-occurence matrix.  Utilize the crosstab method to do this
# print pd.crosstab(df['FTSE'] > df['FTSE'].mean(), df['DAX'] > df['DAX'].mean()) # results show they occur conjointly quite often, therefore you can suppose a strong relationship between these thwo events
## You can examine this graphically as well
import matplotlib.pyplot as plt
plt.scatter(df['FTSE'], df['DAX'], alpha=1.0, color='k')
plt.xlabel("FTSE Index")
plt.ylabel("DAX Index")
plt.show() # the trend is quite marked; therefore we can deduce that x and y are strongly related

## Checking the distribution of the feature by using a historgram to approximate the distribution
plt.hist(df['FTSE'], bins=20)
plt.xlabel("FTSE Index Distribution")
plt.show() # here we see a fairly normal distribution
