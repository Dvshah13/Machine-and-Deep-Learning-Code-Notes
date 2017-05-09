import pandas as pd # importing pandas
import numpy as np # importing numpy for some math functions and to show how pandas and numpy can stack
from pandas import Series, DataFrame # not necessary but will use in script to better explain

# Data Transformation - Includes filtering, cleaning and many other transformation methods as operations that can be performed
# Removing Duplicates:
data = DataFrame({'k1': ['one'] * 3 + ['two'] * 4, 'k2': [1, 1, 2, 3, 3, 4, 4]})
print data
# The DataFrame method duplicated returns a boolean Series indicating whether each row is a duplicate or not:
print data.duplicated()
# Relatedly, drop_duplicates returns a DataFrame where the duplicated array is True
print data.drop_duplicates()
# Both of these methods by default consider all of the columns, alternatively you can specify any subset of them to detect duplicates.  Suppose we had an additional column of values and wanted to filter duplicates only based on the 'k1' column:
data['v1'] = range(7)
print data.drop_duplicates(['k1'])
# duplicated and drop_duplicates by default keep the first observed value combination.  Passing take_last = True will return the last one:
print data.drop_duplicates(['k1', 'k2'], take_last=True)

# Transforming Data using a Function or Mapping
# For many data sets, you may wish to perform some transformation based on the values in an array, Series or column in a DataFrame.
# Hypothetical Example:
data = DataFrame({'food': ['bacon', 'pulled pork', 'bacon', 'Pastrami', 'corned beef', 'Bacon', 'pastrami', 'honey ham', 'nova lox'], 'ounces': [4, 3, 12, 6, 7.5, 8, 3, 5, 6]})
print data
# Suppose you wanted to add a column indicating the type of animal that each food came from.  Write down a mapping of each distinct meat type to the kind of animal:
meat_to_animal = {
    'bacon': 'pig',
    'pulled pork': 'pig',
    'pastrami': 'cow',
    'corned beef': 'cow',
    'honey ham': 'pig',
    'nova lox': 'salmon'
}
# The map method on a Series accepts a function or dict-like object containing a mapping but here we have a small problem in that some of the meats above are capitalized and others are not.  Convert each value to lower case
data['animal'] = data['food'].map(str.lower).map(meat_to_animal)
print data
# We could also have passed a function that does all the work:
print data['food'].map(lambda x: meat_to_animal[x.lower()])
# Using map is a convienient way to perform element-wise transformations and other data cleaning-related operations

# Replacing Values - Filling in missing data with the fillna method can be thought of as a special case of more general value replacement.  While map, as you've seen above can be used to modify a subset of values in an object, replace provides a simpler and more flexible way to do so.
# Consider this series:
data = Series([1., -999., 2., -999., -1000., 3.])
print data
# The -999 values might be sentinel values for missing data.  To replace these NA values that pandas understands we can use replace, producing a new Series:
print data.replace(-999., np.nan)
# If you want to replace multiple values at once, you instead pass a list then the substitute value:
print data.replace([-999., -1000.], np.nan)
# To use a different replacement for each value, pass a list of substitutes:
print data.replace([-999., -1000.], [np.nan, 0])
# The argument passed can also be a dict:
print data.replace({-999.: np.nan, -1000.: 0})

# Renaming Axis Indexes
# Like values in a Series, axis labels can be similarly transformed by a function or mapping of some form to produce new, differently labeled objects.  The axes can also be modified in place without creating a new data structure:
data = DataFrame(np.arange(12).reshape((3, 4)), index=['Ohio', 'Colorado', 'New York'], columns=['one', 'two', 'three', 'four'])
# Like a Series, the axis indexes have a map method:
print data.index.map(str.upper)
# You can assign to index, modifying the DataFrame in place:
data.index = data.index.map(str.upper)
print data
# If you want to create a transformed version of a data set without modifying the original, a useful method is rename:
print data.rename(index=str.title, columns=str.upper) # capitalizes the columns
# Notably, rename can be used in conjunction with a dict-like object providing new values for a subset of the axis labels:
print data.rename(index={'OHIO': 'INDIANA'}, columns={'three': 'peekaboo'})
# Rename saves having to copy the DataFrame manually and assign to its index and columns attributes.  Should you wish to modify a data set in place, pass inplace=True:
_ = data.rename(index={'OHIO': 'INDIANA'}, inplace=True) # Always returns a reference to a DataFrame
print data

# Discretization and Binning - Continuous data is often discretized or otherwise separated into "bins" for analysis.  Suppose you have data about a group of people in a study and you want to group them into discrete age buckets:
ages = [20, 22, 25, 27, 21, 23, 37, 31, 61, 45, 41, 32]
# Let's divide these into bins of 18 to 25, 25 to 35, 35 to 60 and finally 60 and older.
# To do so use cut, a function in Pandas
bins = [18, 25, 35, 60, 100]
age_cat = pd.cut(ages, bins)
print age_cat # The object Pandas returns is a special Categorical object.  You can treat it like an array of strings indicating the bin name; internally it contains a levels array indicating the distinct category names along with a labeling for the ages data in the labels attribute:
print age_cat.codes # used to be labels now codes
# print age_cat.levels
print pd.value_counts(age_cat)
# Consistent with mathematical notation for intervals, a parenthesis means that the side is open while the square bracket means it is closed (inclusive).  Which side is closed can be changed by passing right=False:
print pd.cut(ages, [18, 26, 36, 61, 100], right=False)
# You can also pass your own bin names by passing a list or array to the labels option:
group_names = ['Youth', 'YoungAdult', 'MiddleAged', 'Senior']
print pd.cut(ages, bins, labels=group_names)
# If you pass cut a integer number of bins instead of explicit bin edges, it will compute equal-length bins based on the minimum and maximum values in the data.  Consider the case of some uniformly distributed data chopped into fourths:
data = np.random.rand(20)
print pd.cut(data, 4, precision=2)
# A closely related function, qcut, bins the data based on sample quantiles.  Depending on the distribution of the data, using cut will not usually result in each bin having the same number of data points.  Since qcut uses sample quantiles instead by definition you will obtain roughly equal-size bins:
data = np.random.randn(1000) # Normally distributed
cats = pd.qcut(data, 4) # Cut into quartiles
print cats
print pd.value_counts(cats)
# Similar to cut you can pass your own quantiles (numbers between 0 and 1, inclusive)
print pd.qcut(data, [0, 0.1, 0.5, 0.9, 1.])

# Detecting and Filtering Outliers - Filtering or transforming outliers is largely a matter of applying array operations.  Consider a DataFrame with some normally distributed data:
np.random.seed(12345)
data = DataFrame(np.random.randn(1000, 4))
print data.describe()
# Suppose you wanted to find values in one of the columns exceeding three in magnitude
col = data[3]
print col[np.abs(col) > 3]
# To select all rows having a value exceeding 3 or -3, you can use the any method on a boolean DataFrame
print data[(np.abs(data) > 3).any(1)] # Values can just as easily be set based on these criteria.
# Code to cap values outside the interval -3 to 3:
data[np.abs(data) > 3] = np.sign(data) * 3 # The ufunc np.sign returns an array of 1 and -1 depending on the sign of the value
print data.describe()

# Permutation and Random Sampling - Permuting (randomly reordering) a Series or the rows of a DataFrame is easy to do using the numpy.random.permutation function.  Calling permutation with the length of the axis you want to permute produces an array of integers indicating the new ordering:
df = DataFrame(np.arange(5 * 4).reshape(5, 4))
sampler = np.random.permutation(5)
print sampler
# That array can then be used in ix-based indexing or the take function
print df
print df.take(sampler)
# To select a random subset without replacement, one way is to slice off the first k elements of the array returned by permutation, where k is the desired subset size.  there are much more efficient sampling-without replacement algorithms but this is an easy strategy that uses readily available tools:
print df.take(np.random.permutation(len(df))[:3])
# To generate a sample with replacement, the fastest way is to use np.random.randint to draw random integers:
bag = np.array([5, 7, -1, 6, 4])
sampler = np.random.randint(0, len(bag), size=10)
print sampler
draws = bag.take(sampler)
print draws

# Computing Indicator/Dummy Variables - Another type of transformation for statistical modeling or machine learning applications is converting a categorical variable into a "dummy" or "indicator" matrix.  If a column in a DataFrame has k distinct values, you would derive a matrix or DataFrame containing k columns containing all 1's and 0's.  Pandas has a get_dummies function for doing this, though devising one yourself is not difficult:
df = DataFrame({'key': ['b', 'b', 'a', 'c', 'a', 'b'], 'data1': range(6)})
print pd.get_dummies(df['key'])
# In some cases, you may want to add a prefix to the columns in the indicator DataFrame which can then be merged with the other data.  get_dummies has a prefix argument for doing just this:
dummies = pd.get_dummies(df['key'], prefix='key')
df_with_dummy = df[['data1']].join(dummies)
print df_with_dummy
# If a row in a DataFrame belongs to multiple categories, things are a bit more complicated.
# Using the Movie Lens 1M dataset
mnames = ['movie_id', 'title', 'genres']
movies = pd.read_table('movies.dat', sep='::', header=None, names=mnames)
print movies[:10]
# Adding indicator variables for each genre requires a little bit of wrangling.  First, extract the list of unique genres in the data set (using set.union trick)
genre_iter = (set(x.split('|')) for x in movies.genres)
genres = sorted(set.union(*genre_iter))
# Construct the indicator DataFrame to start with a DataFrame of all zeros:
dummies = DataFrame(np.zeros((len(movies), len(genres))), columns=genres)
# Iterate through each movie and set entries in each row of dummies to 1:
for i, gen in enumerate(movies.genres):
    dummies.ix[i, gen.split('|')] = 1
# Then you can combine this with movies
movies_windic = movies.join(dummies.add_prefix('Genre_'))
print movies_windic.ix[0]
# A useful recipe for statistical applications is to combine get_dummies with a discretization function like cut:
values = np.random.rand(10)
print values
bins = [0, 0.2, 0.4, 0.6, 0.8, 1]
print pd.get_dummies(pd.cut(values, bins))
