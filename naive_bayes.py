# Classification technique based on Bayes theorem with an assumption of independence between predictors.  In simple terms, a Naive Bayes classifier assumes that the presence of a particular feature in a class is unrelated to the presence of any other feature.  Bayes theorem provides a way that we can calculate the probability of a hypothesis given our prior knowledge.  Bayes Theorem is stated as: P(h | d) = (P(d | h) * P(h)) / P(d)
# Naive Bayes is a classification algorithm for binary (two-class) and multi-class classification problems.  The technique is easiest to understand when described using binary or categorical input values.
# It is called naive Bayes or idiot Bayes because the calculation of the probabilities for each hypothesis are simplified to make their calculation tractable.  Rather than attempting to calculate the values of each attribute value P(d1, d2, d3 | h), they are assumed to be conditionally independent given the target value

import csv
import random
import math

# Loading CSV file
def load_csv(filename):
	lines = csv.reader(open(filename, "rb"))
	dataset = list(lines)
	for i in range(len(dataset)):
		dataset[i] = [float(x) for x in dataset[i]]
	return dataset

# split a given dataset into a given split ratio, in our case 0.67 which is common (0.67 for training and 0.33 for test)
def splitDataset(dataset, splitRatio):
	trainSize = int(len(dataset) * splitRatio)
	trainSet = []
	copy = list(dataset)
	while len(trainSet) < trainSize:
		index = random.randrange(len(copy))
		trainSet.append(copy.pop(index))
	return [trainSet, copy]

# separating the training dataset instances by class value so that we can calculate stats for each class. Here we are creating a map of each class value to a list of instances that belong to that class and then sort the entire dataset of instances into the appropriate lists. The function assumes that the last attribute (-1) is the class value. The function returns a map of class values to lists of data instances
def separateByClass(dataset):
	separated = {}
	for i in range(len(dataset)):
		vector = dataset[i]
		if (vector[-1] not in separated):
			separated[vector[-1]] = []
		separated[vector[-1]].append(vector)
	return separated

# calculating the mean of each attribute for a class value. The mean is the central middle/tendency of the data and will used as the middle of our gaussian distribution when calculating probabilities
def mean(numbers):
	return sum(numbers)/float(len(numbers))

# calculating the standard deviation whihc is the square root of the variance. The variance is calculated as the average of the squared differences for each attribute value from the mean. Note we are using the N-1 method, which subtracts 1 from the number of attribute values when calculating the variance
def stdev(numbers):
	avg = mean(numbers)
	variance = sum([pow(x-avg,2) for x in numbers])/float(len(numbers)-1)
	return math.sqrt(variance)

# summerize the data using the zip function which groups the values for each attribute across our data instances into their own lists so that we can compute the mean and standard deviation values for the attribute
def summarize(dataset):
	summaries = [(mean(attribute), stdev(attribute)) for attribute in zip(*dataset)]
	del summaries[-1]
	return summaries

# summerize by class, first we separate our training dataset into instances grouped by class. Then calculate the summaries for each attribute
def summarizeByClass(dataset):
	separated = separateByClass(dataset)
	summaries = {}
	for classValue, instances in separated.iteritems():
		summaries[classValue] = summarize(instances)
	return summaries

# calculate the exponent first, then calculate the main division
def calculateProbability(x, mean, stdev):
	exponent = math.exp(-(math.pow(x-mean,2)/(2*math.pow(stdev,2))))
	return (1 / (math.sqrt(2*math.pi) * stdev)) * exponent

# calculate the probability of an attribute belonging to a class, then combine the probabilities of all of the attribute values for a data instance and come up with a probability of the entire data instance belonging to the class, the result is a map of class values to probabilities
def calculateClassProbabilities(summaries, inputVector):
	probabilities = {}
	for classValue, classSummaries in summaries.iteritems():
		probabilities[classValue] = 1
		for i in range(len(classSummaries)):
			mean, stdev = classSummaries[i]
			x = inputVector[i]
			probabilities[classValue] *= calculateProbability(x, mean, stdev)
	return probabilities

# look for the largest probability and return the associated class
def predict(summaries, inputVector):
	probabilities = calculateClassProbabilities(summaries, inputVector)
	bestLabel, bestProb = None, -1
	for classValue, probability in probabilities.iteritems():
		if bestLabel is None or probability > bestProb:
			bestProb = probability
			bestLabel = classValue
	return bestLabel

# estimate the accuracy of the model by making predictions for each data instance in our test dataset, returns a list of predictions for each test instance
def getPredictions(summaries, testSet):
	predictions = []
	for i in range(len(testSet)):
		result = predict(summaries, testSet[i])
		predictions.append(result)
	return predictions

# calculate accuracy compared with predictions to the class values in the test dataset. Accuracy ratio calculated between 0% and 100%
def getAccuracy(testSet, predictions):
	correct = 0
	for i in range(len(testSet)):
		if testSet[i][-1] == predictions[i]:
			correct += 1
	return (correct/float(len(testSet))) * 100.0

def main():
	filename = 'Data Sets for Code/diabetes.csv'
	splitRatio = 0.67
	dataset = load_csv(filename)
	trainingSet, testSet = splitDataset(dataset, splitRatio)
	print('Split {0} rows into train={1} and test={2} rows').format(len(dataset), len(trainingSet), len(testSet))
	# prepare model
	summaries = summarizeByClass(trainingSet)
	# test model
	predictions = getPredictions(summaries, testSet)
	accuracy = getAccuracy(testSet, predictions)
	print('Accuracy: {0}%').format(accuracy)

main()
