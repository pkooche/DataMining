import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import random

def entropy(y):
	if y.size == 0: return 0
	p = np.unique(y, return_counts = True)[1].astype(float)/len(y)
	return -1 * np.sum(p * np.log2(p+1e-9))

def gini_impurity(y):
	if y.size == 0: return 0
	p = np.unique(y, return_counts = True)[1].astype(float)/len(y)
	return 1 - np.sum(p**2)

def variance(y):
	if y.size == 0: return 0
	return np.var(y)
	

def information_gain(y,mask,func=entropy):
	s1 = np.sum(mask)
	s2 = mask.size - s1
	if (s1 == 0 | s2 == 0): return 0
	return func(y) - s1/float(s1+s2) * func(y[mask]) - s2/float(s1+s2) * func(y[np.logical_not(mask)])


def max_information_gain_split(y,x,func=gini_impurity):
	best_change = None
	split_value = None
	is_numeric = irisX[:,2].dtype.kind not in ['S','b']
	
	for val in np.unique(np.sort(x)):
		mask = x == val
		if(is_numeric): mask = x < val
		change = information_gain(y,mask,func)
		if best_change is None:
			best_change = change
			split_value = val
		elif change > best_change:
			best_change = change
			split_value = val
			
	return {"best_change":best_change,\
			"split_value":split_value,\
			"is_numeric":is_numeric}

def best_feature_split(X,y,func=gini_impurity):
	best_result = None
	best_index = None
	for index in range(X.shape[1]):
		result = max_information_gain_split(y,X[:,index],func)
		if best_result is None:
			best_result = result
			best_index = index
		elif best_result['best_change'] < result['best_change']:
			best_result = result
			best_index = index
	
	best_result['index'] = best_index
	return best_result

def get_best_mask(X,best_feature_dict):
	best_mask = None
	if best_feature_dict['is_numeric']:
		best_mask = X[:,best_feature_dict['index']] < best_feature_dict['split_value']
	else:
		best_mask = X[:,best_feature_dict['index']] == best_feature_dict['split_value']
	return best_mask


class DecisionTreeNode(object):
	
	def __init__(self,\
			X,\
			y,\
			minimize_func,\
			min_information_gain=0.001,\
			max_depth=3,\
			min_leaf_size=20,\
			depth=0):
		self.X = X
		self.y = y
		
		self.minimize_func=minimize_func
		self.min_information_gain=min_information_gain
		self.min_leaf_size = min_leaf_size
		self.max_depth = max_depth
		self.depth = depth
		
		self.best_split = None
		self.left = None
		self.right = None
		self.is_leaf = True
		self.split_description = "root"
		
	def _information_gain(self,mask):
		s1 = np.sum(mask)
		s2 = mask.size - s1
		if (s1 == 0 | s2 == 0): return 0
		return self.minimize_func(self.y) - \
				s1/float(s1+s2) * self.minimize_func(self.y[mask]) - \
				s2/float(s1+s2) * self.minimize_func(self.y[np.logical_not(mask)])
	
	def _max_information_gain_split(self,x):
		best_change = None
		split_value = None
		previous_val = None
		is_numeric = x.dtype.kind not in ['S','b']

		for val in np.unique(np.sort(x)):
			mask = x == val
			if(is_numeric): mask = x < val
			change = self._information_gain(mask)
			s1 = np.sum(mask)
			s2 = mask.size-s1
			#print("elif change::{} > best_change::{} and s1::{} >= self.min_leaf_size::{} and s2::{} >= self.min_leaf_size::{}".format(change,best_change,s1,self.min_leaf_size,s2,self.min_leaf_size))
			# change::0.0 > best_change::None and s1::0 >= self.min_leaf_size::20 and s2::150 >= self.min_leaf_size::20

			if best_change is None and s1 >= self.min_leaf_size and s2 >= self.min_leaf_size:
				best_change = change
				split_value = val
			elif best_change is not None and change > best_change and s1 >= self.min_leaf_size and s2 >= self.min_leaf_size:
				best_change = change
				split_value = np.mean([val,previous_val])
				
			previous_val = val

		return {"best_change":best_change,\
				"split_value":split_value,\
				"is_numeric":is_numeric}
	
	def _best_feature_split(self):
		best_result = None
		best_index = None
		for index in range(self.X.shape[1]):
			result = self._max_information_gain_split(self.X[:,index])
			if result['best_change'] is not None:
				if best_result is None:
					best_result = result
					best_index = index
				elif best_result['best_change'] < result['best_change']:
					best_result = result
					best_index = index
		
		if best_result is not None:
			best_result['index'] = best_index
			self.best_split = best_result
	

	
	def _split_node(self):
		
		if self.depth < self.max_depth :
			
			self._best_feature_split() 
			
			if self.best_split is not None and self.best_split['best_change'] >= self.min_information_gain :
				   
				mask = None
				if self.best_split['is_numeric']:
					mask = self.X[:,self.best_split['index']] < self.best_split['split_value']
				else:
					mask = self.X[:,self.best_split['index']] == self.best_split['split_value']
				
				if(np.sum(mask) >= self.min_leaf_size and (mask.size-np.sum(mask)) >= self.min_leaf_size):
					self.is_leaf = False
					
					self.left = DecisionTreeNode(self.X[mask,:],\
												self.y[mask],\
												self.minimize_func,\
												self.min_information_gain,\
												self.max_depth,\
												self.min_leaf_size,\
												self.depth+1)

					if self.best_split['is_numeric']:
						split_description = 'index ' + str(self.best_split['index']) + " < " + str(self.best_split['split_value']) + " ( " + str(self.X[mask,:].shape[0]) + " )"
						self.left.split_description = str(split_description)
					else:
						split_description = 'index ' + str(self.best_split['index']) + " == " + str(self.best_split['split_value']) + " ( " + str(self.X[mask,:].shape[0]) + " )"
						self.left.split_description = str(split_description)

					self.left._split_node()
					
					
					self.right = DecisionTreeNode(self.X[np.logical_not(mask),:],\
												self.y[np.logical_not(mask)],\
												self.minimize_func,\
												self.min_information_gain,\
												self.max_depth,\
												self.min_leaf_size,\
												self.depth+1)
					
					if self.best_split['is_numeric']:
						split_description = 'index ' + str(self.best_split['index']) + " >= " + str(self.best_split['split_value']) + " ( " + str(self.X[np.logical_not(mask),:].shape[0]) + " )"
						self.right.split_description = str(split_description)
					else:
						split_description = 'index ' + str(self.best_split['index']) + " != " + str(self.best_split['split_value']) + " ( " + str(self.X[np.logical_not(mask),:].shape[0]) + " )"
						self.right.split_description = str(split_description)

				   
					self.right._split_node()
					
		if self.is_leaf:
			if self.minimize_func == variance:
				self.split_description = self.split_description + " : predict - " + str(np.mean(self.y))
			else:
				values, counts = np.unique(self.y,return_counts=True)
				predict = values[np.argmax(counts)]
				self.split_description = self.split_description + " : predict - " + str(predict)
										  
	
	def _predict_row(self,row):
		predict_value = None
		if self.is_leaf:
			if self.minimize_func==variance:
				predict_value = np.mean(self.y)
			else:
				values, counts = np.unique(self.y,return_counts=True)
				predict_value = values[np.argmax(counts)]
		else:
			left = None
			if self.best_split['is_numeric']:
				left = row[self.best_split['index']] < self.best_split['split_value']
			else:
				left = row[self.best_split['index']] == self.best_split['split_value']
				
			if left:
				predict_value = self.left._predict_row(row)
			else:
				predict_value = self.right._predict_row(row)
 
		return predict_value
	
	def predict(self,X):
		print (X)
		return np.apply_along_axis(lambda x: self._predict_row(x),1,X)
	
	def _rep(self,level):
		response = "|->" + self.split_description
		
		if self.left is not None:
			response += "\n"+(2*level+1)*" "+ self.left._rep(level+1)
		if self.right is not None:
			response += "\n"+(2*level+1)*" "+ self.right._rep(level+1)
		
		return response
	
	def __repr__(self):
		return self._rep(0)
		

class DecisionTree(object):
    
    def __init__(self,\
            minimize_func,\
            min_information_gain=0.001,\
            max_depth=3,\
            min_leaf_size=20):
        
        self.root = None
        self.minimize_func = minimize_func
        self.min_information_gain = min_information_gain
        self.max_depth = max_depth
        self.min_leaf_size = min_leaf_size
        
    def fit(self,X,y):
        self.root =  DecisionTreeNode(X,\
                                    y,\
                                    self.minimize_func,\
                                    self.min_information_gain,\
                                    self.max_depth,\
                                    self.min_leaf_size,\
                                    0)
        self.root._split_node()
    
    def predict(self,X):
        return self.root.predict(X)
    
    def __repr__(self):
        return self.root._rep(0)
		


from csv import reader

# Load a CSV file
def load_csv(filename):
	file = open(filename, "r")
	lines = reader(file)
	dataset = list(lines)
	return dataset

# Convert string column to float
def str_column_to_float(dataset, column):
	for row in dataset:
		row[column] = float(row[column].strip())

# Convert string column to integer
def str_column_to_int(dataset, column):
	class_values = [row[column] for row in dataset]
	unique = set(class_values)
	lookup = dict()
	for i, value in enumerate(unique):
		lookup[value] = i
	for row in dataset:
		row[column] = lookup[row[column]]
	return lookup

#Split dataset with ratio
def splitDataset(dataset, splitRatio):
	trainSize = int(len(dataset) * splitRatio)
	trainSet = []
	copy = list(dataset)
	while len(trainSet) < trainSize:
		index = random.randrange(len(copy))
		trainSet.append(copy.pop(index))
	return [trainSet, copy]
	
# Load dataset
filename = 'diabetes-2.csv'
dataset = load_csv(filename)
print('Loaded data file {0} with {1} rows and {2} columns'.format(filename, len(dataset), len(dataset[0])))
#print(dataset[0])
# convert string columns to float
for i in range(len(dataset[0])):
	str_column_to_float(dataset, i)
# convert class column to int
#lookup = str_column_to_int(dataset, 4)
#print(dataset[0])
#print(lookup)
print ("\n" + 50*"*" + "\n")

#build X and Y
splitRatio = 0.67
trainingSet, testSet = splitDataset(dataset, splitRatio)

X = []
Y = []
for row in trainingSet:
	Y.append(row.pop())
	X.append(row)

X = np.array(X) #add Shape to the list object
Y = np.array(Y) #add Shape to the list object
#print (X)
dt_bts = DecisionTree(gini_impurity,min_leaf_size=20,max_depth=3)
dt_bts.fit(X,Y)

TestX = []
TestY = []
for row in testSet:
	TestY.append(row.pop())
	TestX.append(row)

TestX = np.array(TestX) #add Shape to the list object
TestY = np.array(TestY) #add Shape to the list object
#print (TestX)

indexes = []
for x in range(len(testSet)):
	indexes.append(x)
	
real_list = TestY
predicted_list = dt_bts.predict(TestX)

records_limitation_in_plot = 100
plt.plot(indexes[:records_limitation_in_plot], real_list[:records_limitation_in_plot], label='Real Data')
plt.plot(indexes[:records_limitation_in_plot], predicted_list[:records_limitation_in_plot], 'co', label='Predicted')
plt.legend(loc='lower right');
plt.title("Decision Tree", fontsize="10", color="black")
#export_name = 'naive_bayes_{}.png'.format(time.time())
#plt.savefig("./LR/"+export_name, dpi=199)
plt.show()
plt.clf()



print (20*"#" + "\n## Decision Tree: ##\n" + 20*"#")
print (dt_bts)
print ("\n" + 50*"-" + "\n")