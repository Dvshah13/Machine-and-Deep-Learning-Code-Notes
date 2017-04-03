# K-Nearest Neighbors is a simple algorithm that stores all available cases and classifies new cases by a majority vote of its k neighbors.  The case being assigned to the class is most common amongst its K nearest neighbors measured by a distance function.  It is an instance-based algorithms that models the problem using data instances in order to make predictive decisions and because it stores the entire training set no learning is required.  KNN is computationally expensive, variables should be normalized and data must be maintained and preferably curated for noise removal
import csv
import random
import math
import operator

# load data set, in this case it's the iris.data classic classification data set. Here we split the data into a training dataset that KNN can use to make predictions and a test dataset that we can use to evaluate the accuracy of the model. We start by converting the flower measures into numbers from strings.  Then split the data set randomly into train and test, typical split is 2/3 train, 1/3 test
def load_dataset(filename, split, trainingSet=[] , testSet=[]):
	with open(filename, 'rb') as csvfile:
	    lines = csv.reader(csvfile)
	    dataset = list(lines)
	    for x in range(len(dataset)-1):
	        for y in range(4):
	            dataset[x][y] = float(dataset[x][y])
	        if random.random() < split:
	            trainingSet.append(dataset[x])
	        else:
	            testSet.append(dataset[x])

# distance function, most common distance functions are Euclidean, Manhattan, Minkowski and Hamming distance.  First three functions are used for continuous function and fourth one (Hamming) for categorical variables. Euclidian is defined as the square root of the sum of the squared differences between the two arrays of numbers and we also define a length which in our case limits the number of attributes to the first 4
def euclideanDistance(instance1, instance2, length):
	distance = 0
	for x in range(length):
		distance += pow((instance1[x] - instance2[x]), 2)
	return math.sqrt(distance)

# function that returns k most similar neighbors from the training set for a given test instance here we are using the already defined euclideanDistance function
def getNeighbors(trainingSet, testInstance, k):
	distances = []
	length = len(testInstance)-1
	for x in range(len(trainingSet)):
		dist = euclideanDistance(testInstance, trainingSet[x], length)
		distances.append((trainingSet[x], dist))
	distances.sort(key=operator.itemgetter(1))
	neighbors = []
	for x in range(k):
		neighbors.append(distances[x][0])
	return neighbors

# function for getting the majority voted response from a number of neighbors, assumes the class is the last attribute for each neighbor.
def getResponse(neighbors):
	classVotes = {}
	for x in range(len(neighbors)):
		response = neighbors[x][-1]
		if response in classVotes:
			classVotes[response] += 1
		else:
			classVotes[response] = 1
	sortedVotes = sorted(classVotes.iteritems(), key=operator.itemgetter(1), reverse=True)
	return sortedVotes[0][0]

# function evaluating the accuracy of the model by calculating a ratio of the total correct predictions out of all predictions made, called the classification accuracy
def getAccuracy(testSet, predictions):
	correct = 0
	for x in range(len(testSet)):
		if testSet[x][-1] == predictions[x]:
			correct += 1
	return (correct/float(len(testSet))) * 100.0

def main():
	# prepare data
	trainingSet=[]
	testSet=[]
	split = 0.67
	load_dataset('Data Sets for Code/iris.data', split, trainingSet, testSet) # using iris.data data set
	print 'Train set: ' + repr(len(trainingSet))
	print 'Test set: ' + repr(len(testSet))
	# generate predictions
	predictions=[]
	k = 3
	for x in range(len(testSet)):
		neighbors = getNeighbors(trainingSet, testSet[x], k)
		result = getResponse(neighbors)
		predictions.append(result)
		print('> predicted=' + repr(result) + ', actual=' + repr(testSet[x][-1]))
	accuracy = getAccuracy(testSet, predictions)
	print('Accuracy: ' + repr(accuracy) + '%')

main()
