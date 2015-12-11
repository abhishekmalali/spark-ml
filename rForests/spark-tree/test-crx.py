#This script is written to demonstrate the spark implementation of decision trees.
#This script is designed to run and test the crx dataset. The dataset has
#discrete features as well as continuous features.

#Importing spark
import findspark
findspark.init()
import pyspark
sc = pyspark.SparkContext()
import time
#Importing the decisionTree module
from decisionTree import *
from discretize import *

##Loading dataset with continuous features
sample_c = sc.textFile('../data/crx.csv')
rdd_c=sample_c.map(lambda x:x.split()).map(lambda x: x[0].strip("'").split(","))\
            .map(lambda x:[v for v in x])\
            .map(lambda x: (str(x[-1]),[k for k in x[0:-1]]))
columns_c = []
for k in range(1,16):
    columns_c.append('A'+str(k))

#Discretizing the columns which were continuous
train_rdd = discretize_columns(rdd_c,[1,2,7,10,13,14])

#Training the decision tree
start_time = time.time()
tree = createDecisionTree(train_rdd,range(len(columns_c)),columns_c)
print 'Time taken to train tree:',time.time() - start_time

"""
#Testing the accuracy of the decision tree
res_rdd = train_rdd.map(lambda x:(x[0],classify_tree(x[1],columns_c,tree)))
print "Accuracy:", 100*(res_rdd.filter(lambda x:x[0] == x[1]).count())/float(res_rdd.count()),'%'

print "Tree Structure"
print tree
"""
