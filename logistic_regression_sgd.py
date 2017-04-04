# Logistic Regression - used in classification to estimate discrete values (binary values like 0/1, yes/no, true/false) based on a given set of independent variables.  It predicts the probability of an event by fitting data to a logit function.  Also known as logit regression.  Since it predicts the probability, its output lies between 0 to 1.  The logistic function (logit) is also called the sigmoid function.  It’s an S-shaped curve that can take any real-valued number and map it into a value between 0 and 1 but never exactly at those limits.

# Here we're applying stochastic gradient which is a technique that evaluates and updates the coefficients every iteration to minimize the error of a model on our training data.

# Logistic Regression on Diabetes Data set
from csv import reader
from math import exp
from random import seed
from random import randrange

# Loading CSV file, in this example it's the diabetes.csv file located in the Data Sets for Code folder
def load_csv(filename):
	data_set = list()
	with open(filename, 'r') as file:
		csv_reader = reader(file)
		for row in csv_reader:
			if not row:
				continue
			data_set.append(row)
	return data_set

# Convert string column to float, necessary for evaluation
def str_column_to_float(data_set, column):
	for row in data_set:
		row[column] = float(row[column].strip())

# Finding the min and max values for each column and appending them to a list which will be used later
def data_set_min_max(data_set):
	min_max = list()
	for i in range(len(data_set[0])):
		col_values = [row[i] for row in data_set]
		min_value = min(col_values)
		max_value = max(col_values)
		min_max.append([min_value, max_value])
	return min_max

# Rescale data set columns to the range 0-1, good practice because it will allow the algorithm to reach the minimum cost faster if the shape of the cost function is not skewed and distorted.
def normalize_data_set(data_set, min_max):
	for row in data_set:
		for i in range(len(row)):
			row[i] = (row[i] - min_max[i][0]) / (min_max[i][1] - min_max[i][0])

# # The aim in cross-validation is to ensure that every example from the original dataset has the same chance of appearing in the training and testing set. In k-fold cross-validation, the original sample is randomly partitioned into k equal sized subsamples. Of the k subsamples, a single subsample is retained as the validation data for testing the model, and the remaining k − 1 subsamples are used as training data. The cross-validation process is then repeated k times (the folds), with each of the k subsamples used exactly once as the validation data. The k results from the folds can then be averaged to produce a single estimation. The advantage of this method over repeated random sub-sampling is that all observations are used for both training and validation, and each observation is used for validation exactly once.  Using a n_folds of 5 in this function.
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

# Calculate accuracy percentage based on actual vs predicted values.
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
	return 1.0 / (1.0 + exp(-yhat)) # necessary for logistic regression, need data to be between 0 and 1 for classification, equivalent to y = 1.0 / (1.0 + e^(-(b0 + b1 * X1 + b2 * X2)))

# estimate logistic regression coefficients using stochastic gradient descent, Stochastic gradient descent requires two parameters; Learning Rate - Used to limit the amount each coefficient is corrected each time it is updated and Epochs - The number of times to run through the training data while updating the coefficients. These, along with the training data will be the arguments to the function
def coefficients_sgd(train, l_rate, n_epoch):
	coef = [0.0 for i in range(len(train[0]))]
	for epoch in range(n_epoch): # loop over each epoch
		for row in train: # loop over each row in the training data for an epoch.
			yhat = predict(row, coef)
			error = row[-1] - yhat # coefficients are updated based on the error the model made. error is calculated as the difference between the expected output value and the prediction made with the coefficients
			coef[0] = coef[0] + l_rate * error * yhat * (1.0 - yhat)
			for i in range(len(row)-1): # loop over each coefficient and update it for a row in an epoch
				coef[i + 1] = coef[i + 1] + l_rate * error * yhat * (1.0 - yhat) * row[i] # update each coefficient for each row in the training data, each epoch
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
filename = 'Data Sets for Code/diabetes.csv'
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
