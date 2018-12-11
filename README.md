# DataMining




# Comparison of the SVM and Naive bays and desicio tree algorithm 
The ability to produce a diagnosis from large accumulation of patient data has become an
essential tool in health care. Conventional diagnosis methods have failed to handle the
complexity of medical information and the alternative approach of data mining has been adopted.
In the chronic disease Diabetes mellitus, data classification models have been utilized for patient
diagnosis. The determination of an individual’s status; where they have diabetes or not.Data mining has become a formidable tool within the Heath Care. Through the means of data
mining, researchers have been able to uncover new information and insights into specific
medical areas. In the medical research field of the diabetes, data mining algorithms have been
utilized to assist with the diagnosis of the patients. 
We impliment three algorithm to find the best classification algorithm for predicting the diabetes diagnosis .
The data set is download from the Kaggle web site (https://www.kaggle.com/johndasilva/diabetes) 

# Desicion Tree Algorithm 

Decision tree builds classification models in the form of a tree structure. 
It breaks down a dataset into smaller and smaller subsets while at the same time an associated 
decision tree is incrementally developed. The final result is a tree with decision nodes and leaf nodes.

#Framework
 Phyton 3.0 
 online jupyter (https://hub.mybinder.org/user/ipython-ipython-in-depth-hhy7n43i/notebooks/binder/Index.ipynb#)
 library used in our code :
 matplotlib
 numpy 
 pandas 
 random

# Deployment 
first read our data into a painless data frame which is convection named DF and the we also going 
to have to get it into right format for our algorithm . First we should split our data frame in the training and testing 
frame and then we run the decision tree algorithm on the training data to create  our tree and then finally we calculate 
an accuracy to see how good that tree classifies new unknown patient in this way .
 function defined for our coding 
For spliting to train and test data set we define 
   +++++ def train_test_split(df, test_size):
for checking our data is pure or not we define :
+++++def check_purity(data) : 
fod classifying the data we define : 
+++++def classify_data(data):
for se if our data have potential to split we defin the function : 
+++++def get_potential_splits(data):
After we understand that our data have the potential to slplit , we define the split function : 
+++++def split_data(data, split_column, split_value):
After that based an the desicion tree algortihm we should calculate the entropy of each node so we define the entropy function : 
+++++def calculate_entropy(data):
We need to calculate the overal Entropy as well : 
+++++def calculate_overall_entropy(data_below, data_above):
now it's time for defining the decision tree ALG : 
first we should define the features : 
+++++def determine_type_of_feature(df):
after that we defin the algorithm of desicion tree :
+++++def decision_tree_algorithm(df, counter=0, min_samples=2, max_depth=5):
so after deffining the decision tree alg it's time for finfing the accuracy of our alg 
+++++def calculate_accuracy(df, tree): 

**Comment** 

I write these algorithm in 2 ways , the first one is we should remove the names of attributes because they are character and all data all integer so I Have problem for loading the data and having our tree , so I write the algorithm oh desicion_tree_nochr , but I’ll try to find out how can I keep the features name so I write the second code to can have the name of the future by having the data frame and remove the name of the result “Outcome” and put as a “ Label “

df = pd.read_csv("dataset.csv")

df = df.rename(columns={"Outcome": "label"})


# Naive Bayes Algorithm 
Naive Bayes is a kind of classifier which uses the Bayes Theorem. It predicts membership probabilities for each class such as the probability that given record or data point belongs to a particular class.  The class with the highest probability is considered as the most likely class. This is also known as Maximum A Posteriori (MAP).
Naive Bayes classifier assumes that all the features are unrelated to each other. Presence or absence of a feature does not influence the presence or absence of any other feature. We can use Wikipedia example for explaining the logic i.e.,
A fruit may be considered to be an apple if it is red, round, and about 4″ in diameter.  Even if these features depend on each other or upon the existence of the other features, a naive Bayes classifier considers all of these properties to independently contribute to the probability that this fruit is an apple.
In real datasets, we test a hypothesis given multiple evidence(feature). So, calculations become complicated. To simplify the work, the feature independence approach is used to ‘uncouple’ multiple evidence and treat each as an independent one.

naive bayes formula is as follow 

P(c|x)  =  P(x|c)P(c) \ P(x)

P(c|x) is the posterior probability of class (c, target) given predictor (x, attributes).
P(c) is the prior probability of class.
P(x|c) is the likelihood which is the probability of predictor given class.
P(x) is the prior probability of predictor.


#Framework
 Phyton 3.0 
 online jupyter (https://hub.mybinder.org/user/ipython-ipython-in-depth-hhy7n43i/notebooks/binder/Index.ipynb#)

library used in our code :
csv
math
random
matplotlib.pyplot 

# Deployment 
have a training data set of Diabetes corresponding target variable is the patient have diabetes or not . Now, we need to classify diabetes feature follow the below steps to perform it.
Step 1: Convert the data set into a frequency table
Step 2: Create Likelihood table by finding the probabilities
Step 3: Now, use Naive Bayesian equation to calculate the posterior probability for each class.
Naive Bayes uses a similar method to predict the probability of different class based on various attributes. 
Whole process down into the following steps
Step 1:  is handling the data in which we load the data from the CSV file and spread it into training and tested assets
Step 2: is summarizing the data in which summarize the properties in the training dataset so that we can calculate the probabilities and make predictions
Step 3: comes is making a particular prediction we use the summaries of the data set to generate a single prediction
Step 4 : that we generate predictions given a test dataset and a summarized training data sets and
Step 5 :we calculate the accuracy of the predictions made for a test dataset as a percentage correct out of all the predictions made and
Step 6 : finally we tie it together and form our own model of classifier .
function defined for  coding 
When we import the csv file we are converting every element of that data set into float , originally some elements are in string nut we need to convert them into float for our calculation 
+++++dataset = list(lines)
		 for i in range(len(dataset)):
			dataset[i] = [float(x) for x in dataset[i]]

Split the data into training data sets that neighbours can use to make the prediction and test data set that we can use to evaluate the accuracy of the model we need to split the data set randomly into training and testing data set in the ratio of 70% 
+++++def splitDataset(dataset, splitRatio):

Biased model id comprised of summary of the data in the training data set , summary is used while making prediction and it enveloped the mean and standard deviation of each attribute by class value so the first was is to separate the training data set instances by class value so that we calculate statistics for each class we do it by the separate by class function , and assumes that the last attribute is the class value function . 
+++++def separateByClass(dataset):
 
need to calculate the mean  and stand deviation of each attribute for class value 
def mean(numbers):

+++++def stdev(numbers):

Summarizing attributes by class we can pull it all together by first separating ou training data set into instances grouped by class the and calculating the summaries for each attribute 
+++++def summarize(dataset):
+++++def summarizeByClass(dataset):

And we selecting the class with the largest probability as the prediction 
We can decided to 4 tasty gaussian probability density function calculating class probability making a prediction and then estimating the accuracy for calculating the Gaussian probability density function use following function 
+++++def calculateProbability(x, mean, stdev)

Next task is calculating the class properties , calculate the probability of an attribute belonging to a class that can combined the probabilities of all the attributes values for a data instance and come up with probability of the entire data instance belonging to the class 
+++++def calculateClassProbabilities(summaries, inputVector):


For make the first prediction should calculate the probability of the data instance belonging to each class value and looking for the largest probabilities and return the 
associated class 
+++++def predict(summaries, inputVector):

We can estimate the accuracy of the model by making predictions for each data instances in our test data for that we use get prediction function 
+++++def getPredictions(summaries, testSet):

The prediction can be compared to the class values in our test dat set and classification accuracy can be calculated by the accuracy function 
+++++def getAccuracy(testSet, predictions):

***Comment 
Result 
In this Code I show every step of calculation for finding the result of accuracy find the end of the result of the code 
