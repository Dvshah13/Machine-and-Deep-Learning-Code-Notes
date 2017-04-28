## As you start experimenting you realize that both the validation and test results vary as samples are different and the chosen hypothesis is often the best one, but this is not always the case.  Relying on the validation and testing phases of samples brings uncertainty along with a strong reduction of the learning examples for training.
# A solution is to use cross-validation and Scikit Learn offers a complete module for cross-validation and performance evaluation (sklearn.cross_validation)
# By using cross-validation, you just need to separate your data into a training and test set and you'll be able to use the training data for both model optimization and model training.
# The idea behind cross-validation is to divide your training data into a certain number of partitions (called folds) and train your model as many times as the number of partitions, keeping out of training a different partition every time.  Ten folds is quite a common configuration that is recommended.  After each model training, you will test the result on the fold that is left out and store it away.  In the end, you'll have as many results as folds and you can calculate both the average and standard deviation on them.
# The standard deviation will provide a hint on how your model is influenced by the data that is provided for training (the variance of the model, actually), and the mean will provide a fair estimate of its general performance.  Using the mean of the cross-validation results of different models (different due to model type, used selection of the training variables or model's hyper-parameters), you can confidently choose the best performing hypothesis to be tested for general performance.
# Cross validation in code using the digits example:
from sklearn import cross_validation
from sklearn.datasets import load_digits # loading digits dataset
from sklearn import svm
import numpy as np
digits = load_digits()
# print digits.DESCR # prints a description of the dataset
X = digits.data
Y = digits.target
# loading hypothesis
h1 = svm.LinearSVC(C=1.0) # linear SVC
h2 = svm.SVC(kernel='rbf', degree=3, gamma=0.001, C=1.0) # Radial basis SVC
h3 = svm.SVC(kernel='poly', degree=3, C=1.0) # 3rd degree polynomial SVC

choosen_random_state = 1
cv_folds = 10 # you can fiddle with this (maybe try 3,5,20 and see the results)
eval_scoring = 'accuracy' # can also try other metrics such as F1
workers = -1 # this will use all your CPU power
X_train, X_test, Y_train, Y_test = cross_validation.train_test_split(X, Y, test_size = 0.30, random_state = choosen_random_state)
for hypothesis in [h1, h2, h3]:
    scores = cross_validation.cross_val_score(hypothesis, X_train, Y_train, cv = cv_folds, scoring = eval_scoring, n_jobs = workers)
    print "%s -> cross validation accuracy: mean = %0.3f std = %0.3f" % (hypothesis, np.mean(scores), np.std(scores)) # this is the core of the script, the cross_validation.cross_val_score receives the following parameters: A learning algoritms (estimator), A training set of predictors (X), A target variable (Y), The number of cross-validation folds (cv), A scoring function (scoring), The number of CPUs to be used (n_jobs)
# Given the input, the function wraps some complex functions.  It creates n-iterations, training a model of the n-cross-validation in-samples and testing and storing its score on the out-of-sample fold at each iteration.  In the end, the function reports a list of the recorded scores of this kind:
print scores # array of scores
# The main advantage of using cross_val_score resides in its simplicity of usage and the fact that it automatically incorporates all the necessary steps for a correct cross-validation.  When deciding how to split the train sample into folds, if a Y vector is provided, it keeps the same target class label's proportion in each fold as it was in the Y provided

# Using Cross-Validation Iterators - Though the cross_val_score function from the cross_validation module acts as a complete helper function from most of the cross-validation purposes, you may have the necessity to build up your own cross-validation processes.  In this case, the same cross_validation module provides you with a formidable selection of iterators.
# A useful one is cross_validation.KFold, which is quite simple in its functionality.  If n-number of folds are given, it returns n iterations to the indexes of the training and validation sets for the testing of each fold.  Let's say that we have a training set made up of 100 examples and we would like to create a 10-fold cross-validation.
# Set up the iterator
kfolding = cross_validation.KFold(n = 100, n_folds = 10, shuffle = True, random_state = 1) # By using the n parameter, we can instruct the iterator to perform the folding on 100 indexes.  The n_folds specifies the number of folds.  While the shuffle is set to True, it will randomly choose the fold components.  Instead, if it is set to false, the folds will be created with respect to the order of the indexes (so, the first fold will be [0 1 2 3 4 5 6 7 8 9]).  As usual the random_state parameter allows reproducibility of the folds generation
for train_idx, validation_idx in kfolding:
    print train_idx, validation_idx
# during the iterator loop, the indexes for training and validation are provided with respect to your hypothesis for evaluation (let's figure out the h1, the linear SVC).  You just have to select both the X and Y accordingly with the help of fancy indexing:
h1.fit(X[train_idx], Y[train_idx])
print h1.score(X[validation_idx], Y[validation_idx]) # as you can see, a cross-validation iterator provides you with just the index functionality and it is up to you when it comes to using indexes for your scoring evaluation on your hypothesis.  This opens up for you opportunities for elaborate and sophisticated operations.
# Some of the most useful iterator include the following:
# StratifiedKFold - works like KFold but it always returns folds with approximately the same class percentage as the training set.  Instead of the number of cases, as an input parameter, it needs the target variable Y.  It is actually the iterator wrapped by default inthe cross_val_score function that was just seen in the preceding section.
# LeaveOneOut - works like KFold but it returns as a validation set only one observation.  So in the end, the number of folds will be equivalent to the number of examples in the training set.  We recommend that you use this cross-validation approach only when the training set is small, especially is there are less than 100 observations and a k-fold validation would reduce the training set a lot.
# LeavePOut - similar in advantages and limitations to LeaveOneOut but its validation set is made up of P cases.  So, the number of total folds will be the combination of P cases from all the available cases (which actually could be quite a large number as the size of your dataset grows)
# LeaveOneLabelOut - provides a convenient way to cross-validate according to a scheme that you have prepared or computed in advace.  It will act like KFolds but for the the fact the folds will already be labeled and provided to the labels parameter
# LeavePLabelOut - is a variant of LeaveOneLabelOut, here the test folds are made of a number P of labels according to the scheme that you prepare in advance
# Scikit Learns documentations goes into detail how to use (specific parameters required) and various other iterators and is a good place to start
