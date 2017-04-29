## Previously we used a predefined scorer function, for classification there are six measures available and for regression, there are three.  Although they are common measures there are times you might choose to use a different meansure.  For example we may find it useful to use a loss function in order to figure out if even when the classifier is wrong, the right answer is still ranked high in probability (so, the right answer is the second or third option of the algorithm) which can be problematic.
# A work around to use is using in the sklearn.metrics module, using the log_loss function.  All you have to do is wrap that in a way that the GridSearchCV might use it:
from sklearn.metrics import log_loss, make_scorer
Log_Loss = make_scorer(log_loss, greater_is_better = False, needs_proba = True) # created another function (Log_Loss) by calling make_scorer to the log_loss error function from sklearn.metrics.  We also want to point out that we want to minimize this measure (it is a loss, not a score) by setting hte greater_is_better = False.  We also specify that it works with probabilities, not predictions (so, set needs_proba = True).
# Since it works with probabilities, we will use the hp hypothesis, which we just defined in the preceding section, since SVC otherwise won't emit any probability for its predictions:
from sklearn.datasets import load_digits
import numpy as np
digits = load_digits()
X, Y = digits.data, digits.target
# Using two hypotheses
from sklearn import svm
from sklearn import grid_search
search_grid = [  # created a list made of two dictionaries
    {'C': [1, 10, 100, 1000], 'kernel': ['linear']},
    {'C': [1, 10, 100, 1000], 'gamma': [0.001, 0.0001], 'kernel': ['rbf']},
]
h = svm.SVC() # plain SVC that just guesses a label
h_p = svm.SVC(probability = True, random_state = 1) # SVC with the computation of label probabilities (parameter probability=True) with the random_state fixed to 1 for reproducibility of the results.
search_func = grid_search.GridSearchCV(estimator=h_p, param_grid = search_grid, scoring=Log_Loss, n_jobs=-1, iid=False, refit=True, cv=3) # Now, our hyper-parameters are optimized for log loss, not accuracy
search_func.fit(X,Y)
print search_func.best_score_
print search_func.best_params_

# Given the dataset of digits, it could be a challenge to detect a 1 vs 7.  You have to optimize your algorthm to minimize its mistakes on these two numbers
from sklearn.preprocessing import LabelBinarizer
def my_custom_log_loss_func(ground_truth, p_predicitons, penalty=list(), eps=1e-15): # # as a general rule, the first parameter of your function should be the actual answer (ground_truth) and the second should be the predictions or the predicted probabilities (p_predicitons)
    adj_p = np.clip(p_predicitons, eps, 1 - eps)
    lb = LabelBinarizer()
    g = lb.fit_transform(ground_truth)
    if g.shape[1] == 1:
        g = np.append(1 - g, g, axis=1)
    if penalty:
        g[:,penalty] = g[:,penalty] * 2
    summation = np.sum(g * np.log(adj_p))
    return summation * (-1.0/len(ground_truth))

# my_custom_scorer = make_scorer(my_custom_log_loss_func, greater_is_better=False, needs_proba=True, penalty=[4,9]) # here we set the penalty on for highly confusable numbers 4 and 9 (can change it or even leave it empty to check whether the resulting loss will be the same as that of the previous experiment with the sklearn.metrics.log_loss function)
# This new loss function will double log_loss when evaluating the results of the classes of number 4 and 9
search_func = grid_search.GridSearchCV(estimator=h_p, param_grid = search_grid, scoring=my_custom_scorer, n_jobs=-1, iid=False, refit=True, cv=3)
search_func.fit(X,Y)
print search_func.best_score_
print search_func.best_params_

## Reducing Grid Search Runtime - When the data or grid search is space, the procedure may take a long time to compute.  As a different approach, the grid_search module offers RandomizedSearchCV, a procedure that randomly draws a sample of combinations and reports the best combination found.
# Some advantages of RandomizedSearchCV include: 1. You can limit the number of computations.  2. You can obtain a good result or at worst understand where to focus your efforts on in the grid search.  RandomizedSearchCV has the same options as GridSearchCV but: an n_iter parameter, which is the number of random samples and a param_distributions, which has the same functions as that of the param_grid but only accepts deictionaries and works better if you assign distributions as values and not lists of discrete values.  For instance, instead of C: [1, 10, 100, 1000], you can assign a distribution such as C:scipy.stats.expon(scale=100)
# Testing this function with our previous settings:
search_dict = {'kernel': ['linear', 'rbf'], 'C': [1, 10, 100, 1000], 'gamma': [0.001, 0.0001]}
scorer = 'accuracy'
search_func = grid_search.RandomizedSearchCV(estimator=h, param_distributions=search_dict, n_iter=7, scoring=scorer, n_jobs=-1, iid=False, refit=True, cv=10)
search_func.fit(X,Y)
print search_func.best_estimator_
print search_func.best_params_
print search.func.best_score_
# Using just half of the computations (7 draws against 14 trials with the exhaustive grid search), it found an eqivalent solution.  Even without a complete overview of all combinations, a good sample can prompt you to look for just the RBF kernel and for certain C and gamma ranges, limiting a following grid search to a limited portion of the potential search space
