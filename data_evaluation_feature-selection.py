## Irrelevant and redundant features may play a role in the lack of interpretability of the resulting model, long training times and most importantly, overfitting and poor generalization.
# Overfitting is related to the ratio of the number of observations and the variables available in your dataset.  When the variables are many compared to the observations, your learning algorithm will have more chance of ending up with some local optimization or the fitting of some spurious noise due to the correlation between variables.
# Apart from dimensionality reduction, which requires you to transform the data, feature selection can be the solution to the aforementioned problems.  It simplifies high dimensional structures by choosing the most predictive set of variables, that is, it picks the features that work well together, even if some of them are not such good predictors on an independent level.
# Scikit-learn offers a wide range of feature selection methods including: univariate selection, recursive elimination, randomized logistic regression/stability selection, L1-based feature selection, tree-based feature selection.
# Note: univariate and recursive elimination can be found in the feature_selection module.  The others are a by-product of specific machine learning algorithms.

# Univariate Selection - Here we intend to select single variables that are associated the most with your target variable according to a statistical test.  There are three available tests: 1. the f_regression object uses an F-test and a p-value according to the ratio of explained variance against the unexplained one in a linear regression of the variable with the target.  This is useful only for regression problems.  2. The f_classif object is an Anova F test that can be used when dealing with classification problems.  3. The Chi2 object is a chi-squared test, which is suitable when the target is a classification and variables are count or binary data (they should be positive)
# All the tests have a score and p-value.  Higher scores and p-values indicate that the variable is associated and is therefore useful to the target.  The test do not take into account instances where the variable is a duplicate or is highly correlated to another variable.  It is therefore mostly useful to rule out the not-so-useful variables than to highlight the most useful ones.
# In order to automate the procedure, there are also some selection routines available: SelectKBest, based on the score of the test, takes the k best variables. SelectPercentile, based on the score of the test, takes the top percentile of performing variables.  Based on the p-values of the tests, SelectFpr (false positive rate test), SelectFdr (false discovery rate test),and SelectFwe (family wise error rate procedure)
# You can also create your own selection procedure with the GenericUnivariateSelect function using the score_func parameter, which takes predictors and the target and returns a score and p-value based on your favorite statistical test.  The great advantage offered by these functions is that they offer a series of methods to select the variables (fit) and later on reduce (transform) all the sets to the best variables.  In our example, we use the .get_support() method in order to get a Boolean indexing from both the Chi2, and f_classif tests on the top 25 percent predictive variables.  We then decide on the variables selected by both the tests:
from sklearn.datasets import make_classification
X, Y = make_classification(n_samples=800, n_features=100, n_informative=25, n_redundant=0, random_state=101) # creates a dataset of 800 cases and 100 features.  The most important variables are a quarter of the total
from sklearn.feature_selection import SelectPercentile
from sklearn.preprocessing import Binarizer, scale
Xbin = Binarizer().fit_transform(scale(X))
# If using chi2, input X must be non-negative: X must contain booleans or frequencies
# Hence the choice to binarize after the normalization of the variable if above the average
Selector_chi2 = SelectPercentile(chi2, percentile=25).fit(Xbin, Y)
Selector_f_classif = SelectPercentile(f_classif, percentile=25).fit(X,Y)
chi_scores = Selector_chi2.get_support()
f_classif_scores = Selector_f_classif.get_support()
selected = chi_scores & f_classif_scores # use the bitwise and operator
# the final selected variable contains a Boolean vector, pointing out 21 predictive variables that are pointed out by both tests

# Recursive Elimination - The problem with univariate selection is the likelihood of selecting a subset containing redundant information, whereas our interest is to get a minimum set that works with our predictor algorithm.  A recursive elimination in this case could help provide the answer.
from sklearn.cross_validation import train_test_split
X,Y = make_classification(n_samples=100, n_features=100, n_informative=5, n_redundant=2, random_state=101) # small dataset with quite a large number of features, shows a problem of p > n, where p is the number of variables and n is the number of observations
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.30, random_state=101)

from sklearn.linear_model import LogisticRegression
classifier = LogisticRegression(random_state=101)
classifier.fit(X_train, Y_train)
print "In-Sample Accuracy: %0.3f" % classifier.score(X_train, Y_train)
print "Out-of-Sample Accuracy: %0.3f" % classifier.score(X_test, Y_test)
# There are surely some informative variables in the datast but the noise provided by the others may fool the learning algorithm in assigning the correct coefficients to the correct features.  This reflects in high (perfect 1.0 in our case) in-sample accuracy.  However in a poorer test accuracy, the RFECV class, provided with a learning algorithm and instructions about the scoring/loss function and the cross-validation procedure, starts fitting an initial model on all variables and calculates a score based on cross-validation.  At this point, RFECV starts pruning the variables until it reaches a set of variables where the cross-validated score start decreasing (whereas by pruning the score should have stayed stable or increased)
from sklearn.feature_selection import RFECV
selector = RFECV(estimator=classifier, step=1, cv=10, scoring='accuracy')
selector.fit(X_train, Y_train)
print("Optimal Number of Features: %d" % selector.n_features_) # here we got optimal features of 17, which means out of 100 variables the RFECV ended up selecting 17.  We can check this result on the test set after transforming both the training and test set in order to reflect the variable pruning:
X_train_s = selector.transform(X_train)
X_test_s = selector.transform(X_test)
classifier.fit(X_train_s, Y_train)
print "Out-of-Sample accuracy: %0.3f" % classifier.score(X_test_s, Y_test)
# As a general rule when you notice a large discrepancy between the training result (based on cross-validation, not the in-sample score) and the out-of-sample results, recursive selection can help you achieve better performance from your learning algorithms by pointing out some of the most important variables.

## Stability and L1-based Selection - However, though effective, recursive elimination is actually a greedy algorithm. While pruning, it opts for certain selections, potentially excluding many others.  That's a good way to reduce an NP-hard-problem, such as an exhaustive search among possible sets, into a more manageable one.  There's another way to solve the problems using all the variables at hand conjointly.  Some algorithms use regularization to limit the weight of the coefficients, thus preventing overfitting and the selection of the most relevant variables without losing predictive power.  In particular, the regularization L1 (lasso) is well-known for the creation of sparse selection of variables' coefficients since it pushes many variables to the 0 value according to the set strength of regularization.
# An example to clarify the usage of the logistic regression classifier and the sythetic dataset that we used for recursive elimination
from sklearn.svm import LinearSVC
classfier = LogisticRegression(C=0.1, penalty='l1', random_state=101) # the smaller C the fewer features selected
classifier.fit(X_train, Y_train)
print "Out of Sample Accuracy: %0.3f" % classifier.score(X_test, Y_test)
# The out-of-sample accuracy is better than the previous one that was obtained by using the greedy approach.  The scret is the penalty=l1 and the C value that was assigned when initializing the LogisticRegression class.  Since C is the main indredient of L1-based selection, it is important to choose it correctly.  This can be done using cross-validation but an easier and more effective way is using stability selection.
# Stability Selection uses L1 regularization even under the default values (though you may change them in order to improve the results) because it verifies its results by subsampling, that is, by recalculating the regularization process a large number of times using a randomly chosen part of the training dataset.  The final result excludes all the variables that often had their coefficient estimated to zero.  Only if a variable has most of the time a non zero coefficient will the variable be considered stable to the dataset and feature set variations, and important to be included in the model (hence the name stability selection)
# Implementing the selection approach
from sklearn.linear_model import RandomizedLogisticRegression
selector = RandomizedLogisticRegression(n_resampling=300, random_state=101)
selector.fit(X_train, Y_train)
print "Variables selected: %i" % sum(selector.get_support() != 0)
X_train_s = selector.transform(X_train)
X_test_s = selector.transform(X_test)
classifier.fit(X_train_s, Y_train)
print "Out-of-Sample Accuracy: %0.3f" % classifier.score(X_test_s, Y_test)
# As a matter of fact, we obtained results that were similar to that of the L1-based selection by just using the default parameters of the RandomizedLogisticRegression class.
# this algorithm works fine, it is reliable and out of the box (no parameters to tweak unless you want to try lowering the C values in order to speed it up).  It's suggested to set the n_resampling parameter to a large number so that your computer can handle in  reasonable amount of time.  If you want to select for a regression problem, you should use the RandomizedLasso class instead.
