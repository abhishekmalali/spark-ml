#This script is written to demonstrate the spark implementation of decision trees.
#This script is designed to run and test the balance scale dataset. The dataset has
#only discrete features.

#Importing spark
import findspark
findspark.init()
import pyspark
sc = pyspark.SparkContext()
import time
#Importing the decisionTree module
from decisionTree import *
from discretize import *

#Loading the balance-scale data
sample = sc.textFile('../data/balance-scale.csv')
data = sample.map(lambda x:x.split()).map(lambda x: x[0].strip("'").split(","))\
            .map(lambda x:[v for v in x])\
            .map(lambda x: (str(x[0]),[int(k) for k in x[1:]]))
columns = ['Left-Weight','Left-Distance','Right-Weight','Right-Distance']

#Training the decision tree
start_time = time.time()
tree = createDecisionTree(data,range(len(columns)),columns)
print 'Time taken to train tree:',time.time() - start_time

#Testing the accuracy of the decision tree
res_rdd = data.map(lambda x:(x[0],classify_tree(x[1],columns,tree)))
print "Accuracy:", 100*(res_rdd.filter(lambda x:x[0] == x[1]).count())/float(res_rdd.count()),'%'

print "Tree Structure"
print tree
