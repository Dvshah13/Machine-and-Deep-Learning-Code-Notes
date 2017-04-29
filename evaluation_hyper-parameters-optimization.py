## A machine learning hypothesis is not only determined by the learning algorithm but also by its hyper-parameters (the parameters of the algortihm that have to be a priori fixed and which cannot be learned during the training process) and the selection of variables to be used to achieve the best learned parameters.
# You can extend the cross-validation approach to find the best hyper-parameters that are able to generalize to our test set.  An example using the handwritten digits dataset
from sklearn.datasets import load_digits
import numpy as np
digits = load_digits()
X, Y = digits.data, digits.target
# use the SVM as our learning algorithm
from sklearn import svm
# Using two hypotheses
h = svm.SVC() # plain SVC that just guesses a label
h_p = svm.SVC(probability = True, random_state = 1) # SVC with the computation of label probabilities (parameter probability=True) with the random_state fixed to 1 for reproducibility of the results.
# SVC is useful for all the loss metrics that require a probability and not a prediction, such as AUC, to evaluate the machine learning estimator's performance.
# Now we can import the grid_search module and set the list of hyper-parameters that we want to test by cross-validation
# Utilize the GridSearchCV function which will automatically search for the best parameters according to a search schedule and score the results with respect to a predifined or custom scoring function:
from sklearn import grid_search # import module
search_grid = [  # created a list made of two dictionaries
    {'C': [1, 10, 100, 1000], 'kernel': ['linear']},
    {'C': [1, 10, 100, 1000], 'gamma': [0.001, 0.0001], 'kernel': ['rbf']},
]
scorer = 'accuracy' # set the scorer variable using a string (accuracy), the scorer is a string that we chose from a range of possible ones (scikit learn documentation has a list of others)
# Using predefined values just requires you to pick your evaluation metric from the list (there are some for classification and regression, and there are also some for clustering) and use the string by plugging it directly, or by using a string variable into the GridSearchCV function.
# GridSearchCV also accepts a parameter called param_grid which can be a dictionary containing, as keys, an indication of all the hyper-parameters to be changed and, as values of the dictionary keys, lists of parameters to be tested. So if you want to test the performances of your hypothesis with respect to the hyper-parameter C, you can create a dictionary like this:
{'C': [1, 10, 100, 1000]}
# Alternatively, according to your preference, you can use a specialized NumPy function to generate numbers that are evenly spaced on a log scale:
{'C' : np.logspace(start=-2, stop=3, num=6, base=10.0)}
# You can therefore enumerate all the possible parameters' values and test all their combinations.  However, you can also stack different dictionaries, with each dictionary containing only a portion of the parameters that can be tested together.  For example, when working with SVC, the kernel set to linear automatically excludes the gamma parameter.  Combining it with the linear kernel would be in fact a waste of computational power since it would not have any effect on the learning process
search_func = grid_search.GridSearchCV(estimator=h, param_grid=search_grid, scoring=scorer, n_jobs=-1, iid=False, refit=True, cv=10) # estimator is our hypothesis we defined (h), param_grid is the dictionary we defined as search_grid, scoring is the scorer we defined as 'accuray', n_jobs=-1 just forces the function to use all the processors available on the computer, refit=True - so that the function fits the whole training set using the best estimator's parameters.  Now we just need to apply the search_func.predict() method to fresh data in order to obtain new predictions, the cv parameter is set to 10 foles (you can go for a smaller number, trading off speed with accuracy of testing)
# the iid paramter is set to False - this parameter decides how to compute the error measure with respect to the classes.  If the classes are balanced, setting iid won't have much effect.  However if unbalanced, by default (iid=True) will make the classes with more examples weigh more on the global error. Instead, iid=False means that all the classes should be considered the same.  Since we wanted SVC to recognize every handwritten number from 0 to 9 no matter how many examples were given for each of them, we decided that setting iid parameter to False was the right choice.  Depending on your project, you may decide to set it to the default of True
print search_func.best_estimator_
print search_func.best_params_
print search_func.best_score_
