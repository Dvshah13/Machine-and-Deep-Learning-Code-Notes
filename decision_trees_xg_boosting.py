# XG Boosting is an application of gradient boosting to increase speed and performance.  Boosting in general is an ensemble learning algorithm which combines the prediction of several base estimators in order to improve robustness over a single estimator. It combines multiple weak or average predictors to a build strong predictor.

# Here we are importing the xgboost module with XGBClassifier

# XGBoost on train_xgboost.csv data set
from pandas import read_csv
from xgboost import XGBClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder
import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot
import numpy
# load data
data = read_csv('Data Sets for Code/train_xgboost.csv')
dataset = data.values
# split data into X and y, here we're separate the columns (attributes or features) of the dataset into input patterns (X) and output patterns (Y).
X = dataset[:,0:94]
y = dataset[:,94]
# encode string class values as integers
label_encoded_y = LabelEncoder().fit_transform(y) # encoded here to normalize data from 0-1
# building model and setting some initial params then doing the grid search
model = XGBClassifier()
n_estimators = [50, 100, 150, 200] # number of boosted trees to fit
max_depth = [2, 4, 6, 8]  # should typically be between 2-10
print(max_depth)
param_grid = dict(max_depth=max_depth, n_estimators=n_estimators) # Dictionary with parameters names (string) as keys and lists of parameter settings to try as values, or a list of such dictionaries, in which case the grids spanned by each dictionary in the list are explored. Grid search provided by GridSearchCV exhaustively generates candidates from a grid of parameter values specified with the param_grid parameter.
kfold = StratifiedKFold(n_splits=10, shuffle=True, random_state=7) # Provides train/test indices to split data in train/test sets.  This cross-validation object is a variation of KFold that returns stratified folds. The folds are made by preserving the percentage of samples for each class
grid_search = GridSearchCV(model, param_grid, scoring="neg_log_loss", n_jobs=-1, cv=kfold, verbose=1) # search over specified parameter values for an estimator, here we take our model, param_grid and the attributes scoring = scorer callable object / function, n_jobs = number of jobs to run in parallel, cv = determines the cross-validation splitting strategy, using k-fold, verbose = controls the verbosity: the higher, the more messages
grid_result = grid_search.fit(X, label_encoded_y) # here we are trying to get get the best parameter combination, X is from the input patterns and label_encoded_y is the y values normalized from 0-1
# summarizing results
print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_)) # retrieving the best score and params of grid
means = grid_result.cv_results_['mean_test_score']
stds = grid_result.cv_results_['std_test_score']
params = grid_result.cv_results_['params']
for mean, stdev, param in zip(means, stds, params):
	print("%f (%f) with: %r" % (mean, stdev, param))
# plotting results
scores = numpy.array(means).reshape(len(max_depth), len(n_estimators))
for i, value in enumerate(max_depth):
    pyplot.plot(n_estimators, scores[i], label='depth: ' + str(value))
pyplot.legend()
pyplot.xlabel('n_estimators')
pyplot.ylabel('Log Loss')
pyplot.savefig('n_estimators_vs_max_depth.png')
