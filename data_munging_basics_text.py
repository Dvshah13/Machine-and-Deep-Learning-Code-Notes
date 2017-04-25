## Data muning basics with text data, using the 20 newsgroup dataset
from sklearn.datasets import fetch_20newsgroups ## this automatically downloads the dataset and places it in a default directory
categories = ['sci.med', 'sci.space']
twenty_sci_news = fetch_20newsgroups(categories=categories)
# You can now query the dataset object by asking for the location of the files, their content and label.  Located in the .filenames, .data, .target attributes respectively
print twenty_sci_news.data[0]  # prints out the content of the file
print twenty_sci_news.filenames # prints filenames
print twenty_sci_news.target[0] # the target is categorical but represented as an integer (0 for sci.med, 1 for sci.space)
print twenty_sci_news.target_names[twenty_sci_news.target[0]] # prints the name of the category

## Transform the body of dataset into a series of words to make it easier to deal with, for each document, the number of times a specific word appears in the body will be counted
from sklearn.feature_extraction.text import CountVectorizer
count_vect = CountVectorizer() # instantiate countvectorizer object
word_count = count_vect.fit_transform(twenty_sci_news.data) # call method to count the terms in each document and produce a feature vector for each of them (fit_transform)
print word_count.shape # the first value is the number of observations in the dataset (number of documents), while the second number is the number of features (the number of unique words in the dataset)

print word_count[0] # looking at first document, we get a sparse matrix where only nonzero elements are stored
# to see the direct correspondence to the words use this:
word_list = count_vect.get_feature_names()
for n in word_count[0].indices:
    print "Word:", word_list[n], "appears", word_count[0,n], "times"

## Compute the frequency, it's a measure that you can compare across differently sized datasets.  It gives an idea whether a word is a stop word (that is, common word such as a, an, the, is) or a rare, unique one.  Typically, these terms are the most important because they're able to characterize an instance and the features based on these words, which are very discrimative in the learning process.
from sklearn.feature_extraction.text import TfidfVectorizer
tf_vect = TfidfVectorizer(use_idf=False, norm='l1')
word_freq = tf_vect.fit_transform(twenty_sci_news.data)
word_list = tf_vect.get_feature_names()
for n in word_freq[0].indices:
    print "Word:", word_list[n], "has frequency", word_freq[0,n]
# the sum of the frequencies is 1 (approximately), this is because we chose the l1 norm.  In this specific case, the word frequency is a probability distribution function, sometimes it's nice to increase the difference between rare and common words, in that case use l2 norm to normalize the feature vector

## An even more effective way to vectorize the text data is by using Tfidf, you can multiply the term frequency of the words that compose a document by the inverse document frequency of the word itself (that is, in the number of documents it appears, if logarithmically scaled).  This is very handy to highlight words that effectively describe each document, and is a powerful discriminative element among the dataset
from sklearn.feature_extraction.text import TfidfVectorizer
tfidf_vect = TfidfVectorizer() # default: use_idf=True
word_tfidf = tfidf_vect.fit_transform(twenty_sci_news.data)
word_list = tfidf_vect.get_feature_names()
for n in word_tfidf[0].indices:
    print "Word:", word_list[n], "has tdidf", word_tfidf[0,n]
# the most characterizing words are the ones that have the highest tfidf score.  This means that their term frequency within the document is hight, whereas they're pretty rare in the remaining documents.

## Evaluating a pair of words.  Called bigrams or n-grams, the presence or absence of a word as well as its neighbors matters (that is, the words near it and their disposition)
from sklearn.feature_extraction.text import CountVectorizer
text_1 = "i love to code data science in python"
text_2 = "data science can be challenging but fun at the same time"
documents = [text_1, text_2]
print documents
# using the default of 1-grams
count_vect_1_grams = CountVectorizer(ngram_range=(1,1), stop_words=[], min_df=1)
word_count = count_vect_1_grams.fit_transform(documents)
word_list = count_vect_1_grams.get_feature_names()
print "Word list = ", word_list
print "text_1 is decribed with", [word_list[n] + "(" + str(word_count[0,n]) + ")" for n in word_count[0].indices]

## Bi-count vectorizer
count_vect_1_grams = CountVectorizer(ngram_range=(2,2))
word_count = count_vect_1_grams.fit_transform(documents)
word_list = count_vect_1_grams.get_feature_names()
print "Word list = ", word_list
print "text_1 is decribed with", [word_list[n] + "(" + str(word_count[0,n]) + ")" for n in word_count[0].indices]

## A uni-gram (1-gram) and bi-gram count vectorizer
count_vect_1_grams = CountVectorizer(ngram_range=(1,2))
word_count = count_vect_1_grams.fit_transform(documents)
word_list = count_vect_1_grams.get_feature_names()
print "Word list = ", word_list
print "text_1 is decribed with", [word_list[n] + "(" + str(word_count[0,n]) + ")" for n in word_count[0].indices]
# This intuitively composes the first and second approach, in this example we used a CountVectorizer but this approach is very common with TfidfVectorizer.  The number of features explodes exponentially when you use n-grams

## If you have too many features (the dictionary may be too rich, there may be too many n-grams or the computer may just be limted), you can use a trick that lowers the complexity of the problem (be sure and evaluate the trade-off performance/trade-off complexity).  It's common to use the hashing trick where many words (or n-grams) are hashed and their hashes collide (which makes a bucket of words).  Buckets are sets of sematically unrelated words but with colliding hashes.  With HashingVectorizer(), you can decide the number of buckets of words you want. The resulting matrix, reflects your setting.
from sklearn.feature_extraction.text import HashingVectorizer
hash_vect = HashingVectorizer(n_features=1000)
word_hashed = hash_vect.fit_transform(twenty_sci_news.data)
print word_hashed.shape
# You can't invert the hashing process (since it's a digest operation).  Therefore, after this transformation, you will have to work on the hashed features as they are
