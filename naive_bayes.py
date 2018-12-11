
import csv
import math
import random

import matplotlib.pyplot as plt

#Handle data
def loadCsv(filename):
	lines = csv.reader(open(filename, "r"))
	dataset = list(lines)
	for i in range(len(dataset)):
		dataset[i] = [float(x) for x in dataset[i]]
	return dataset

#Test handling data
""" 
filename = 'pima-indians-diabetes.data.csv'
SomeDataset = loadCsv(filename)
print("Loaded data file {0:s} with {1:5d} rows".format(filename,len(SomeDataset)))
"""

#Split dataset with ratio
def splitDataset(dataset, splitRatio):
	trainSize = int(len(dataset) * splitRatio)
	trainSet = []
	copy = list(dataset)
	while len(trainSet) < trainSize:
		index = random.randrange(len(copy))
		trainSet.append(copy.pop(index))
	return [trainSet, copy]

#Test splitting data
"""
dataset = [[1], [2], [3], [4], [5]]
splitRatio = 0.67
train, test = splitDataset(dataset, splitRatio)
print('Split {0} rows into train with {1} and test with {2}'.format(len(dataset),train,test))
"""

#Separate by Class
def separateByClass(dataset):
	separated = {}
	for i in range(len(dataset)):
		vector = dataset[i]
		if (vector[-1] not in separated):
			separated[vector[-1]] = []
		separated[vector[-1]].append(vector)
	return separated

#Test separating by class
"""
dataset = [[1,20,1],[2,21,0],[3,22,1]]
separated = separateByClass(dataset)
print('Separated instances: {0}'.format(separated))
"""

#Calculate Mean
def mean(numbers):
	return sum(numbers)/float(len(numbers))

def stdev(numbers):
	avg = mean(numbers)
	variance = sum([pow(x-avg,2) for x in numbers])/float(len(numbers)-1)
	return math.sqrt(variance)

#Test stdev & mean calculation
"""
numbers = [1,2,3,4,5]
print('Summary of {0}: mean={1}, stdev={2}'.format(numbers, mean(numbers), stdev(numbers)))
"""

#Summarize Dataset
def summarize(dataset):
	summaries = [(mean(attribute), stdev(attribute)) for attribute in zip(*dataset)]
	del summaries[-1]
	return summaries

#Test summarizing data
"""
dataset = [[1,20,0], [2,21,1], [3,22,0]]
summary = summarize(dataset)
print('Attribute summaries: {0}'.format(summary))
"""

#Summarize attributes by class
def summarizeByClass(dataset):
	separated = separateByClass(dataset)
	summaries = {}
	for classValue, instances in separated.items():
		summaries[classValue] = summarize(instances)
	
	print ("\nSummarize attributes by class =>\nsummaries =>\n{}".format(summaries))
	return summaries

#Test summarizing attributes
"""
dataset = [[1,20,1], [2,21,0], [3,22,1], [4,22,0]]
summary = summarizeByClass(dataset)
print('Summary by class value: {0}'.format(summary))
"""

#Calculate Gaussian Probability Density Function
def calculateProbability(x, mean, stdev):
	exponent = math.exp(-(math.pow(x-mean,2)/(2*math.pow(stdev,2))))
	print ("\nCalculate Gaussian Probability Density Function =>\nX: {}, mean: {}, stdev: {}\nProbability: {}".format(x, mean, stdev, (1/(math.sqrt(2*math.pi)*stdev))*exponent))
	return (1/(math.sqrt(2*math.pi)*stdev))*exponent

#Testing Gaussing PDF
"""
x = 71.5
mean = 73
stdev = 6.2
probability = calculateProbability(x,mean,stdev)
print('Probability of belonging to this class: {0}'.format(probability))
"""

#Calculate Class Probabilities
def calculateClassProbabilities(summaries, inputVector):
	probabilities = {}
	for classValue, classSummaries in summaries.items():
		probabilities[classValue] = 1
		for i in range(len(classSummaries)):
			mean, stdev = classSummaries[i]
			x = inputVector[i]
			probabilities[classValue] *= calculateProbability(x, mean, stdev)
		print("\nCalculate Class Probabilities\nsummaries: {}, inputVector: {}\nClassProbabilities: {}".format(summaries, inputVector, probabilities))
		return probabilities

#Testing Class Probability calculation
"""
summaries = {0:[(1, 0.5)], 1:[(20, 5.0)]}
inputVector = [1.1, '?']
probabilities = calculateClassProbabilities(summaries, inputVector)
print('Probabilities for each class: {0}'.format(probabilities))
"""

#Make a prediction
def predict(summaries, inputVector):
	probabilities = calculateClassProbabilities(summaries, inputVector)
	bestLabel, bestProb = None, -1
	for classValue, probability in probabilities.items():
		if bestLabel is None or probability > bestProb:
			bestProb = probability
			bestLabel = classValue
	print("\nMake a prediction\nsummaries: {} \ninputVector: {} \nbestLabel: {}".format(summaries, inputVector, bestLabel))
	return bestLabel

#Test prediction
"""
summaries = {'A':[(1, 0.5)], 'B':[(20, 5.0)]}
inputVector = [1.1, '?']
result = predict(summaries, inputVector)
print('Prediction: {0}'.format(result))
"""

#Get predictions

def getPredictions(summaries, testSet):
	predictions = []
	for i in range(len(testSet)):
		result = predict(summaries, testSet[i])
		predictions.append(result)
	return predictions

#Test predictions
"""
summaries = {'A':[(1, 0.5)], 'B':[(20, 5.0)]}
testSet = [[1.1,'?'], [19.1,'?']]
predictions = getPredictions(summaries, testSet)
print('Predictions: {0}',format(predictions))
"""

#Get Accuracy
def getAccuracy(testSet, predictions):
	correct = 0
	indexes = []
	real_list = []
	predicted_list = []
	
	for x in range(len(testSet)):
		indexes.append(x)
		real_list.append(testSet[x][-1])
		predicted_list.append(predictions[x])
		
		if testSet[x][-1] == predictions[x]:
			correct += 1
			
	#print(predicted_list)
	records_limitation_in_plot = 100
	plt.plot(indexes[:records_limitation_in_plot], real_list[:records_limitation_in_plot], label='Real Data')
	plt.plot(indexes[:records_limitation_in_plot], predicted_list[:records_limitation_in_plot], 'co', label='Predicted')
	plt.legend(loc='lower right');
	plt.title("Naive Bayes", fontsize="10", color="black")
	#export_name = 'naive_bayes_{}.png'.format(time.time())
	#plt.savefig("./LR/"+export_name, dpi=199)
	plt.show()
	plt.clf()
	
	return (correct/float(len(testSet)))*100.0

#Test accuracy
"""
testSet = [[1,1,1,'a'], [2,2,2,'a'], [3,3,3,'b']]
predictions = ['a', 'a', 'a']
accuracy = getAccuracy(testSet, predictions)
print('Accuracy: {0}'.format(accuracy))
"""
def main():
	filename = 'diabetes-2.csv'
	splitRatio = 0.67
	dataset = loadCsv(filename)
	trainingSet, testSet = splitDataset(dataset, splitRatio)
	print('Split {0} rows into train = {1} and test = {2} rows'.format(len(dataset),len(trainingSet),len(testSet)))
	#prepare model
	summaries = summarizeByClass(trainingSet)
	#test model
	predictions = getPredictions(summaries, testSet)
	accuracy = getAccuracy(testSet, predictions)
	print('Accuracy: {0}%'.format(accuracy))
	
	'''indexes = []
	for index in range(len(testSet)):
		indexes.append(index)
	'''
	#print(testSet[0])
	#print(testSet[0][-1])
	#print (testSet.to_dict(orient='list')[-1])
	
	#plt.plot(indexes, testSet.to_dict(orient='list')['Final_Target'], label='True data')
	'''
	plt.plot(Sorted_X_test[:100].index.values, y_test[:100].to_dict(orient='list')['Final_Target'], label='True data')
	plt.plot(Sorted_X_test[:100].index.values, y_pred[:100], 'co', label='LR | {}'.format(params))
	#plt.plot(X_test[1::2], pred_lm[1::2], 'mo', label='Linear Reg')
	plt.legend(loc='lower right');
	plt.title(scores_as_title, fontsize="10", color="black")
	export_name = 'LR_{}.png'.format(time.time())
	plt.savefig("./LR/"+export_name, dpi=199)
	#plt.show()
	plt.clf()
	'''

main()