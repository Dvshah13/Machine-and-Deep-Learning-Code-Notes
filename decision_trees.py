# Decision Trees/CART, supervised learning algorithm mostly used for classification purposes but also works for both categorical and continuous dependent variables. In this algorithm, you are trying to split the population into two or more homogeneous sets.  This is done based on the most significant attributes/independent variables to make as distinct groups as possible also known as splitting the data based on the feature that results in the largest information gain.  Although mostly used for classification purposes, you can also use it for regression predictive modeling problems. The CART algorithm provides a foundation for important algorithms like bagged decision trees, random forests and boosted decisions trees.  The representation for the CART model is a binary tree.  Each root node represents a single input variable (x) and a split point on that variable (assuming variable is numeric). The leaf nodes of the tree contains an output variable (y) which is used to make a prediction

# Decision Tree/Classification and Regression Tree(CART) Algorithm implementation using banknote database from UCI Machine Learning Repository
from random import seed
from random import randrange
from csv import reader

# Load CSV file
def load_csv(filename):
	file = open(filename, "rb")
	lines = reader(file)
	dataset = list(lines)
	return dataset

# Convert string column to float, necessary for evaluation for categorical data
def str_column_to_float(dataset, column):
	for row in dataset:
		row[column] = float(row[column].strip())

# The aim in cross-validation is to ensure that every example from the original dataset has the same chance of appearing in the training and testing set.  In k-fold cross-validation, the original sample is randomly partitioned into k equal sized subsamples. Of the k subsamples, a single subsample is retained as the validation data for testing the model, and the remaining k − 1 subsamples are used as training data. The cross-validation process is then repeated k times (the folds), with each of the k subsamples used exactly once as the validation data. The k results from the folds can then be averaged to produce a single estimation. The advantage of this method over repeated random sub-sampling is that all observations are used for both training and validation, and each observation is used for validation exactly once.  Using a n_folds of 5 in this function.
def cross_validation_split(dataset, n_folds):
	dataset_split = list()
	dataset_copy = list(dataset)
	fold_size = int(len(dataset) / n_folds)
	for i in range(n_folds):
		fold = list()
		while len(fold) < fold_size:
			index = randrange(len(dataset_copy))
			fold.append(dataset_copy.pop(index))
		dataset_split.append(fold)
	return dataset_split

# Calculate accuracy percentage based on actual vs predicted
def accuracy_metric(actual, predicted):
	correct = 0
	for i in range(len(actual)):
		if actual[i] == predicted[i]:
			correct += 1
	return correct / float(len(actual)) * 100.0

# Evaluate an algorithm using a cross validation split
def evaluate_algorithm(dataset, algorithm, n_folds, *args):  # using *args since it will allow you pass an arbitrary number of arguments to your function.
	folds = cross_validation_split(dataset, n_folds)
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

# Split a dataset based on an attribute and an attribute value
def test_split(index, value, dataset):
	left, right = list(), list()
	for row in dataset:
		if row[index] < value:
			left.append(row)
		else:
			right.append(row)
	return left, right

# Calculate the Gini index for a split dataset which is representative of the cost function as seen in many ML algorithms.  For classification the Gini cost function is used which provides an indication of how pure the leaf nodes are (how mixed the training data assigned to each node is).
# G = sum (pk * (1 - pk)).  G is the Gini cost over all classes, pk are the number of training instances with class k in the rectangle of interest.
# For splitting some common measures include the gini index, entropy and classification error.
def gini_index(groups, class_values):
	gini = 0.0
	for class_value in class_values:
		for group in groups:
			size = len(group)
			if size == 0:
				continue
			proportion = [row[-1] for row in group].count(class_value) / float(size)
			gini += (proportion * (1.0 - proportion))
	return gini

# Select the best split point for a dataset again maximizing information gain
def get_split(dataset):
	class_values = list(set(row[-1] for row in dataset))
	b_index, b_value, b_score, b_groups = 999, 999, 999, None
	for index in range(len(dataset[0])-1):
		for row in dataset:
			groups = test_split(index, row[index], dataset)
			gini = gini_index(groups, class_values)
			if gini < b_score:
				b_index, b_value, b_score, b_groups = index, row[index], gini, groups
	return {'index':b_index, 'value':b_value, 'groups':b_groups}

# Create a terminal node value, this is where there are no further decision nodes. Terminal nodes depict the final outcomes of the decision making process.
def terminal_node(group):
	outcomes = [row[-1] for row in group]
	return max(set(outcomes), key=outcomes.count)

# Create child splits for a node or make terminal again making splits till we reach the terminal nodes
def split(node, max_depth, min_size, depth):
	left, right = node['groups']
	del(node['groups'])
	# check for a no split
	if not left or not right:
		node['left'] = node['right'] = terminal_node(left + right)
		return
	# check for max depth
	if depth >= max_depth:
		node['left'], node['right'] = terminal_node(left), terminal_node(right)
		return
	# process left child
	if len(left) <= min_size:
		node['left'] = terminal_node(left)
	else:
		node['left'] = get_split(left)
		split(node['left'], max_depth, min_size, depth+1)
	# process right child
	if len(right) <= min_size:
		node['right'] = terminal_node(right)
	else:
		node['right'] = get_split(right)
		split(node['right'], max_depth, min_size, depth+1)

# Building the decision tree
def build_tree(train, max_depth, min_size):
	root = get_split(train)
	split(root, max_depth, min_size, 1)
	return root

# Make a prediction with a decision tree, using isinstance to check if object is of object type specified otherwise returns false and else in our model
def predict(node, row):
	if row[node['index']] < node['value']:
		if isinstance(node['left'], dict):
			return predict(node['left'], row)
		else:
			return node['left']
	else:
		if isinstance(node['right'], dict):
			return predict(node['right'], row)
		else:
			return node['right']

# Classification and Regression Tree Algorithm, returning predictions from tree that is built
def decision_tree(train, test, max_depth, min_size):
	tree = build_tree(train, max_depth, min_size)
	predictions = list()
	for row in test:
		prediction = predict(tree, row)
		predictions.append(prediction)
	return(predictions)

# Testing done on signals_sonar_classify data set
seed(1) # generate random number
# load and prepare data
filename = 'Data Sets for Code/signals_sonar_classify.csv'
dataset = load_csv(filename)
# convert string attributes to integers
for i in range(len(dataset[0])):
	str_column_to_float(dataset, i)
# evaluate algorithm
n_folds = 5
max_depth = 5
min_size = 10
scores = evaluate_algorithm(dataset, decision_tree, n_folds, max_depth, min_size)
print('Scores: %s' % scores)
print('Mean Accuracy: %.3f%%' % (sum(scores)/float(len(scores))))
