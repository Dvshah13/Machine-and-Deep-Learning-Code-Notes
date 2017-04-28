## Testing and Validation - process by which find the correct learning process to apply in order to achieve the best generalizable model for prediction.
# Example of best practices to follow:
from sklearn.datasets import load_digits # loading digits dataset
digits = load_digits()
# print digits.DESCR # prints a description of the dataset
X = digits.data
Y = digits.target
# These digits are acutally stored as a vector (resulting from the flattening of each 8 X 8 image) of 64 numeric values from 0 to 16, representing greyscale tonality for each pixel
# print X[0] # outputs an array of values from 0 to 16 as mentioned above
# We will upload three different machine learning hypotheses and three support vector machines for classification.  They will also be useful for our practical example:
from sklearn import svm
h1 = svm.LinearSVC(C=1.0) # linear SVC
h2 = svm.SVC(kernel='rbf', degree=3, gamma=0.001, C=1.0) # Radial basis SVC
h3 = svm.SVC(kernel='poly', degree=3, C=1.0) # 3rd degree polynomial SVC
# As a first experiment, let's fit the linear SVC to our data and verify the results
h1.fit(X,Y) # fits a model using the x-array in order to correctly predict one of the 10 classes indicated by the Y vector.
# print h1.score(X,Y) # calling the .score method and specifying the same predictors (the X-array), the method evaluates the performance in terms of mean accuracy with respect to the true values given by the Y vector.  The result is about 99.05 percent accurate in predicting the correct digit.
# this is the in-sample performance, it's important to note that if the model is overtrained or too complext it will yield good results with the in-sample data it has trained on but with fresh data it will fail.  This is due to overfitting the training data.  Therefore to have a proper estimate of the predictive performance of our hypothesis, we need to test it on some fresh data where there is no memorization effect.
# many methods to deal with memorization, you can increase the number of examples, use a simpler algorithm or use regularization to penalize extremely complex models and force the algorithm to underweight or even exclude some variables.
# many times fresh data is not available, that's why we divide the data into a training (70-80 percent) and test set (20-30 percent remaining).  The split should be completely random, taking into account any possible unbalanced class distribution:
from sklearn import cross_validation
chosen_random_state = 1 # can change to another int
X_train, X_test, Y_train, Y_test = cross_validation.train_test_split(X,Y, test_size=0.30, random_state=chosen_random_state) # initial data is randomly split into two sets thanks to cross_validation.train_test_split() function on the basis of test_size.  The split if governed by the random state which assures that the operation is reproducible at different times and on different computers (even if using different operating systems)
print "X train shape %s, X test shape %s, \nY train shape %s, Y test shape %s" % (X_train.shape, X_test.shape, Y_train.shape, Y_test.shape)
h1.fit(X_train, Y_train)
print h1.score(X_test, Y_test) # returns the mean accuracy on the given test data and labels
# you can modify the chosen_random_state can get different accuracy scores, signaling that the test set is also not an absolute measure of performance and that it should be used.
# We can get a biased performance estimation from the test set if we either choose (after various trials with random_state) a test set that can confirm our hypothesis or start using the test set as a reference to take decisions in regard to the learning process (for example, selecting the best hypothesis that fits a certain test sample).
# The resulting performance would look better but would not be a representation of the real performance of the machine learning system we've build, therefore we have to choose between multiple hypotheses (commonly done in data science) after fitting each of them onto the training data, we need a data sample that can be used to compare their performances, and it cannot be the test set (because of the reason mentioned previously)
# Thus a correct approach would be to use a validation set.  It's suggested that when you initally split the data; 60 percent of the initial data should be reserved for the training set, 20 percent for the validation set and 20 percent for the test set.  Here is how to implement this in code:
chosen_random_state = 1
X_train, X_validation_test, Y_train, Y_validation_test = cross_validation.train_test_split(X, Y, test_size = .40, random_state = chosen_random_state) # dividing the data into training set and testing/validation set
X_validation, X_test, Y_validation, Y_test = cross_validation.train_test_split(X_validation_test, Y_validation_test, test_size = .50, random_state = chosen_random_state) # dividing the testing/validation set into two parts testing and validation
print "X train shape, %s, X validation shape %s, X test shape %s, \nY train shape %s, Y validation shape %s, Y test shape %s\n" % (X_train.shape, X_validation.shape, X_test.shape, Y_train.shape, Y_validation.shape, Y_test.shape)
for hypothesis in [h1, h2, h3]:
    hypothesis.fit(X_train, Y_train)
    print "%s -> validation mean accuracy = %0.3f" % (hypothesis, hypothesis.score(X_validation, Y_validation))
    h2.fit(X_train, Y_train)
    print "\n%s -> test mean accuracy = %0.3f" % (h2, h2.score(X_test, Y_test))\
# As reported by the output, the training set is now made up of 1078 cases (60 percent of the total cases).  In order to divide the data into three parts - training, validation, and test - at first, the data is divided using the cross_validation.train_test_split function between the train and test/validation set.  Then the test/validation dataset is split into two parts using the same function. Each hypothesis, after being trained is tested against the validation set.  Our results showed the RBF kernel by obtaining a mean accuracy of 0.992 is the best model according to the validation set.  Having decided to use this model, its performance was further evaluated on the test set, resulting in an accuracy of 0.978.
# Since the test's accuracy is different from that of the validation one, is the chosen hypothesis really the best one?  It's good practice to try and run the code in the cell multiple times (ideally, running the code 30 times shoudl ensure statistical significance), each time changing the chosen_random_state value.  The same learning procedure will therefore be validated with respect to different samples.
