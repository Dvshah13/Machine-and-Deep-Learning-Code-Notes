import pandas as pd # importing pandas
import numpy as np # importing numpy for some math functions and to show how pandas and numpy can stack
from pandas import Series, DataFrame # not necessary but will use in script to better explain

# Pandas has a number of features for reading tabular data as a DataFrame object.  Most common are read_csv and read_table
# Type inference is one of the most important features of these functions; you don't have to specify which columns are numeric, integer, boolean or string.

# JSON Data - Several built in libraries for reading and writing JSON.
# To convert a JSON string to Python form, use json.loads
import json
# obj = {"name": "Wes", "places_lived": ["United States", "Spain", "Germany"], "pet": null, "siblings": [{"name": "Scott", "age": 25, "pet": "Zuko"}, {"name": "Katie", "age": 33, "pet": "Cisco"}]}
# result = json.loads(obj)
# print result
# json.dumps on the otherhand converts a Python object back to JSON:
# asjson = json.dumps(result)
# You can convert a JSON object or list of objects to a DataFrame or some other data structure for analysis.  Conveniently, you can pass a list of JSON objects to the DataFrame constructor and select a subset of the data fields:
# siblings = DataFrame(result['siblings'], columns['name', 'age'])
# print siblings

# XML and HTML: Web Scraping - Python has many libraries for reading and writing data in HTML and XML, lxml is one that has consistently strong performance in parsing very large files.
# lxml for html
from lxml.html import parse
from urllib2 import urlopen

parsed = parse(urlopen('http://finance.yahoo.com/q/op?s=AAPL+Options'))
doc = parsed.getroot()
# Using this object, you can extract all HTML tags of a particular type, such as table tags containing the data of interest.
# If you wanted to get a list of every URL linked to in the document; links are a tags in HTML. Using the document root's findall method along with an XPath (a means of expressing "queries" on the document):
links = doc.findall('.//a')
print links[15:20] # returns objects representing HTML elements
lnk = links[28]
print lnk # returns object representing HTML elements
print lnk.get('href') # used to get url of HTML element
print lnk.text_content() # displays text of HTML element
# Thus, getting a list of all URLs in the document is a matter of writing this list comprehension
urls = [lnk.get('href') for lnk in doc.findall('.//a')]
print urls[-10:]
# Finding the right tables in the document can be a matter of trial and error; some websites make it easier by giving a table of interest and id attribute.  I determined that these were the two tables containing  the call data and put data, respectively
tables = doc.findall('.//table')
calls = tables[9]
puts = tables[13]
# Each table has a header row followed by each of the data rows
rows = calls.findall('.//tr')
# For the header as well as the data rows, we want to extract the text from each cell; in the case of the header these are th cells and td cells for the data:
def unpack(row, kind='td'):
    elts = row.findall('.//%s' % kind)
    return [val.text_content() for val in elts]

# Now you can combine all these steps together to convert this data into a DataFrame.  Since the numerical data is still in string format, we want to convert some but perhaps not all of the columns to floating point format.  You could do this by hand but pandas has a class called TextParser that is used internally in the read_csv and other parsing functions to do the appropriate automatic type conversion:
from pandas.io.parsers import TextParser

def parse_options_data(table):
    rows = table.findall('.//tr')
    header = unpack(rows[0], kind='th')
    data = [unpack(r) for r in rows[1:]]
    return TextParser(data, names=header).get_chunk()
# Invoke this parsing function on the lxml table objects and get DataFrame results:
call_data = parse_options_data(calls)
put_data = parse_options_data(puts)
print call_data[:10]

# Parsing XML with lxml.objectify
# Many sites keep performance data in a set of XML files
from lxml import objectify

path = 'Your path.xml'
parsed = objectify.parse(open(path))
root = parsed.getroot()
# root.INDICATOR return a generator yielding each <INDICATOR> XML element.  For each record, we can populate a dict of tag names (like YTD_ACUTAL) to data values (excluding a few tags):
data = []
skip_fields = ['Put in whatever fields you want to skip']
for elt in root.INDICATOR:
    el_data = {}
    for child in elt.getchildren():
        if child.tag in skip_fields:
            continue
        el_data[child.tag] = child.pyval
    data.append(el_data)
# Convert this list of dicts into a DataFrame:
perf = DataFrame(data)
print perf
# XML data can get much more complicated than this example.  Each tag can have metadata too, consider an HTML link tag which is also valid XML:
from StringIO import StringIO
tag = '<a href="http://www.google.com">Google</a>'

root = objectify.parse(StringIO(tag)).getroot()
# You can now access any of the fields (like HREF) in the tag or the link text:
print root
print root.get('href')
print root.text

# Binary Data Formats - One of the easiest ways to store data efficiently in binary format is using Python's built-in pickle serialization.  Conveniently, pandas objects all have a save method which writes the data to disk as a pickle:
df = pd.read_csv('Data Sets for Code/faithful.csv')
df.save('Data Sets for Code/frame_pickle') # writing the data to disk as pickle
df.load('Data Sets for Code/frame_pickle') # you can the data back into Python with pandas.load
# Note: Pickle is only recommended as a short-term storage format.  The problem is that it is hard to guarantee that the format will be stable over time; an object pickled today may not unpickle with a later version of a library.

# HDF5 Format - Number of tools that facilitate efficiently reading and writing large amounts of scientific data in binary format on disk.  HDF5 is a popular industry-grade library, which is a C library which can interface with many other languages.  HDF stands for hierarchical data format.  Each HDF5 file contains an internal file system-like node structure enabling you to store multiple datasets and supporting metadata.  HDF5 supports on-the-fly compression with a variety of compressors, enabling data with repeated patterns to be stored more efficiently.  For very large datasets that don't fit into memory, HDF5 is a good choice as you can efficiently read and write small sections of much larger arrays.
# Two interfaces to the HDF5 library in Python, PyTable and h5py, h5py provides a direct but high-level interface to the HDF5 API, while PyTable abstracts many of the details of HDF5 to provide multiple flexible data containers, table indexing, querying capability, and some support for out-of-core computations.
# Pandas has a minimal dict-like HDF5Store class, which uses PyTables to store pandas objects
store = pd.HDF5Store('mydata.h5')
store['obj1'] = frame
store['obj1_col'] = frame['a']
print store
# Objects contained in the HDF5 file can be retrieved in a dict-like fashion:
print store['obj1']

# Interacting with HTML and Web APIs - Many sites have public APIs providing data feeds via JSON or some other format.  One method to use with python is the requests package
import requests
url = 'http://search.twitter.com/search.json?q=python%20pandas'
resp = requests.get(url)
print resp
# The Response object's text attribute contains the content of the GET query.  Many web APIs will return a JSON string that must be loaded into a Python object
import json
data = json.loads(resp.text)
print data.keys() # Outputs in json format the keys
# The results field in the response contains a list of tweets, each of which is represented as a Python dict
# We can make a list of the tweet fields of interest then pass the results list to a DataFrame
tweet_fields = ['created_at', 'from_user', 'id', 'text']
tweets = DataFrame(data['results'], columns=tweet_fields)
print tweets # Shows we've pass data to DataFrame format
print tweets.ix # Each row in the DataFrame now has the extracted data from each tweet

# Databases - You can load information with SQL-based relational databases (SQL Server, PostGres, MySQL) or non-SQL (mongodb)
# Loading data from SQL into a DataFrame is fairly straightforward, and pandas has some functions to simplify the process.
# You can use an in-memory SQLite database using Python's built-in sqlite3 driver:
import sqlite3
query = """
CREATE TABLE test
(a VARCHAR(20), b VARCHAR(20), c REAL, d INTEGER
); """
con = sqlite3.connect(':memory:')
con.execute(query)
con.commit()
# Insert a few rows of data
data = [('Atlanta', 'Georgia', 1.25, 6), ('Tallahassee', 'Florida', 2.6, 3), ('Sacramento', 'California', 1.7, 5)]
stmt = "INSERT INTO test VALUES(?, ?, ?, ?)"
con.executemany(stmt, data)
con.commit()
# You can pass the list of tuples to the DataFrame constructor, but you also need the column names, contained in the cursor's description attribute
print cursor.description
print DataFrame(rows, columns=zip(*cursor.description)[0])
# This is a bit of munging that you'd rather not repeat each time you query the database.  Pandas has a read_frame function in its pandas.io.sql module that simplifies the process.  Just pass the select statement and the connection object
import pandas.io.sql as sql
print sql.read_frame('select * from test', con)
# Storing and Loading Data in MongoDB
import pymongo
con = pymongo.Connection('localhost', port=27017)
# Documents stored in MongoDB are found in collections inside databases.  Each running instance of the MongoDB server can have multiple databases and each database can have multiple collections.  An example, to store the Twitter API data from earlier in the chapter.
# Access the (currently empty) tweets collection:
tweets = con.db.tweets
# Load the list of tweets and write each of them to the collection using tweets.save (writes the Python dict to MongoDB):
import requests, json
url = 'http://search.twitter.com/search.json?q=python%20pandas'
data = json.loads(requests.get(url).text)
for tweet in data['results']:
    tweets.save(tweet)
# If you wanted all my tweets from collection, you can query the collection with the following:
cursor = tweets.find({'from_user': 'wesmckinn'}) # the cursor returned is an iterator that yields each document as a dict.
# You can convert this into a DataFrame, optionally extracting a subset of the data fields in each tweet:
tweet_fields = ['created_at', 'from_user', 'id', 'text']
result = DataFrame(list(cursor), columns=tweet_fields) 
