# Linear Regression With Stochastic Gradient Descent
# Linear Regression is at its core used to estimate real values based on a continuous variable.  You aim to establish a relationship between independent and dependent variables by fitting a best fit line.  This best fit line is known as the regression line and is represented by a linear equation: Y = a * X + b
# Stochastic Gradient Descent useful for very large data sets.  This is a variation of gradient descent where the update to the coefficients is performed for each training instance, rather than at the end of the batch of instances like in gradient descent.
from csv import reader
from math import sqrt
from random import seed
from random import randrange

# Load a CSV file, data set used
def load_csv(filename):
	data_set = list()
	with open(filename, 'r') as file:
		csv_reader = reader(file)
		for row in csv_reader:
			if not row:
				continue
			data_set.append(row)
	return data_set

# Convert string column to float, conversion necessary to treat data as numerical
def str_column_to_float(data_set, column):
	for row in data_set:
		row[column] = float(row[column].strip())

# Iterate through data set and append min_max with the min and max to be used later
def data_set_min_max(data_set):
	min_max = list()
	for i in range(len(data_set[0])):
		col_values = [row[i] for row in data_set]
		min_value = min(col_values)
		max_value = max(col_values)
		min_max.append([min_value, max_value])
	return min_max

# Normalizing the data set, this is not necessary but good practice because it will allow the algorithm to reach the minimum cost faster if the shape of the cost function is not skewed and distorted.  Here we return values from 0,1
def normalize_data_set(data_set, min_max):
	for row in data_set:
		for i in range(len(row)):
			row[i] = (row[i] - min_max[i][0]) / (min_max[i][1] - min_max[i][0])

# Split a data set into k folds. In k-fold cross-validation, the original sample is randomly partitioned into k equal sized subsamples. Of the k subsamples, a single subsample is retained as the validation data for testing the model, and the remaining k − 1 subsamples are used as training data. The cross-validation process is then repeated k times (the folds), with each of the k subsamples used exactly once as the validation data. The k results from the folds can then be averaged to produce a single estimation. The advantage of this method over repeated random sub-sampling is that all observations are used for both training and validation, and each observation is used for validation exactly once.  Using a n_folds of 5 in this function.
def cross_validation_split(data_set, n_folds):
	data_set_split = list()
	data_set_copy = list(data_set)
	fold_size = int(len(data_set) / n_folds)
	for i in range(n_folds):
		fold = list()
		while len(fold) < fold_size:
			index = randrange(len(data_set_copy))
			fold.append(data_set_copy.pop(index))
		data_set_split.append(fold)
	return data_set_split

# Calculate root mean squared error, RSME is the square root of the variance of the residuals or the absolute measure of fit. It indicates the absolute fit of the model to the data–how close the observed data points are to the model’s predicted values.
def rmse_metric(actual, predicted):
	sum_error = 0.0
	for i in range(len(actual)):
		prediction_error = predicted[i] - actual[i]
		sum_error += (prediction_error ** 2)
	mean_error = sum_error / float(len(actual))
	return sqrt(mean_error)

# Evaluate an algorithm using a cross validation split
def evaluate_algorithm(data_set, algorithm, n_folds, *args):
	folds = cross_validation_split(data_set, n_folds)
	scores = list()
	for fold in folds:
		train_set = list(folds)
		train_set.remove(fold)
		train_set = sum(train_set, [])
		test_set = list()
		for row in fold:
			row_copy = list(row)
			test_set.append(row_copy)
			row_copy[-1] = None
		predicted = algorithm(train_set, test_set, *args)
		actual = [row[-1] for row in fold]
		rmse = rmse_metric(actual, predicted)
		scores.append(rmse)
	return scores

# Make a prediction with coefficients
def predict(row, coefficients):
	yhat = coefficients[0]
	for i in range(len(row)-1):
		yhat += coefficients[i + 1] * row[i]
	return yhat

# Estimate linear regression coefficients using stochastic gradient descent
def coefficients_sgd(train, l_rate, n_epoch):
	coef = [0.0 for i in range(len(train[0]))]
	for epoch in range(n_epoch):
		for row in train:
			yhat = predict(row, coef)
			error = yhat - row[-1]
			coef[0] = coef[0] - l_rate * error
			for i in range(len(row)-1):
				coef[i + 1] = coef[i + 1] - l_rate * error * row[i]
			# print(l_rate, n_epoch, error)
	return coef

# Linear Regression Algorithm With Stochastic Gradient Descent
def linear_regression_sgd(train, test, l_rate, n_epoch):
	predictions = list()
	coef = coefficients_sgd(train, l_rate, n_epoch)
	for row in test:
		yhat = predict(row, coef)
		predictions.append(yhat)
	return(predictions)

# Linear Regression on wine quality data set
seed(1)
# load and prepare data
filename = 'Data Sets for Code/red_wine_quality-lin_reg.csv'
data_set = load_csv(filename)
for i in range(len(data_set[0])):
	str_column_to_float(data_set, i)
# normalize
min_max = data_set_min_max(data_set)
normalize_data_set(data_set, min_max)
# evaluate algorithm
n_folds = 5
l_rate = 0.01 # keep it small, can use 0.001 as well.
n_epoch = 50  # number of epochs to train, using 50 epochs or exposures of the coefficients to the entire training data set.
scores = evaluate_algorithm(data_set, linear_regression_sgd, n_folds, l_rate, n_epoch)
print('Scores: %s' % scores)
print('Mean RMSE: %.3f' % (sum(scores)/float(len(scores))))
