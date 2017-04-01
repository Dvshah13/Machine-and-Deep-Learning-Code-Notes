# Logistic Regression on Diabetes Data set
from csv import reader
from math import exp
from random import seed
from random import randrange

# Load a CSV file
def load_csv(filename):
	data_set = list()
	with open(filename, 'r') as file:
		csv_reader = reader(file)
		for row in csv_reader:
			if not row:
				continue
			data_set.append(row)
	return data_set

# Convert string column to float
def str_column_to_float(data_set, column):
	for row in data_set:
		row[column] = float(row[column].strip())

# Find the min and max values for each column
def data_set_min_max(data_set):
	min_max = list()
	for i in range(len(data_set[0])):
		col_values = [row[i] for row in data_set]
		min_value = min(col_values)
		max_value = max(col_values)
		min_max.append([min_value, max_value])
	return min_max

# Rescale data set columns to the range 0-1
def normalize_data_set(data_set, min_max):
	for row in data_set:
		for i in range(len(row)):
			row[i] = (row[i] - min_max[i][0]) / (min_max[i][1] - min_max[i][0])

# Split a data set into k folds
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

# Calculate accuracy percentage
def accuracy_metric(actual, predicted):
	correct = 0
	for i in range(len(actual)):
		if actual[i] == predicted[i]:
			correct += 1
	return correct / float(len(actual)) * 100.0

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
		accuracy = accuracy_metric(actual, predicted)
		scores.append(accuracy)
	return scores

# Make a prediction with coefficients
def predict(row, coefficients):
	yhat = coefficients[0]
	for i in range(len(row)-1):
		yhat += coefficients[i + 1] * row[i]
	return 1.0 / (1.0 + exp(-yhat))

# Estimate logistic regression coefficients using stochastic gradient descent
def coefficients_sgd(train, l_rate, n_epoch):
	coef = [0.0 for i in range(len(train[0]))]
	for epoch in range(n_epoch):
		for row in train:
			yhat = predict(row, coef)
			error = row[-1] - yhat
			coef[0] = coef[0] + l_rate * error * yhat * (1.0 - yhat)
			for i in range(len(row)-1):
				coef[i + 1] = coef[i + 1] + l_rate * error * yhat * (1.0 - yhat) * row[i]
	return coef

# Linear Regression Algorithm With Stochastic Gradient Descent
def logistic_regression(train, test, l_rate, n_epoch):
	predictions = list()
	coef = coefficients_sgd(train, l_rate, n_epoch)
	for row in test:
		yhat = predict(row, coef)
		yhat = round(yhat)
		predictions.append(yhat)
	return(predictions)

# Test the logistic regression algorithm on the diabetes data set
seed(1)
# load and prepare data
filename = 'Data Sets for Code/diabetes-log_reg.csv'
data_set = load_csv(filename)
for i in range(len(data_set[0])):
	str_column_to_float(data_set, i)
# normalize
min_max = data_set_min_max(data_set)
normalize_data_set(data_set, min_max)
# evaluate algorithm
n_folds = 5
l_rate = 0.1
n_epoch = 100
scores = evaluate_algorithm(data_set, logistic_regression, n_folds, l_rate, n_epoch)
print('Scores: %s' % scores)
print('Mean Accuracy: %.3f%%' % (sum(scores)/float(len(scores))))
