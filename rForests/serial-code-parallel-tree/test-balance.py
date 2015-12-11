# Starting spark
#Only use locally
import findspark
findspark.init()
import pyspark
sc = pyspark.SparkContext()
import numpy as np
import time
import csv
import itertools

#Making necessary imports
from decisionTree import *
from infoGain import *
from bootstrap import *
from multiParallelTree import *

#Loading the balance dataset
sample = []
with open('../data/balance-scale.csv', 'rb') as csvfile:
    reader = csv.reader(csvfile, delimiter=',')
    for row in reader:
        sample.append((str(row[0]),row[1:]))

columns = ['Left-Weight','Left-Distance','Right-Weight','Right-Distance']
#Creating an RDD with bootstrapped data
# n is number of trees in our forest
start_time = time.time()
ntrees = 500
res,train_data = create_Forest(sc, sample, ntrees, columns, 3, discrete_column_ids=[], n_bins=10)
print "Time taken to train forest: ", time.time() - start_time
#Checking accuracy
rdd = sc.parallelize(sample)
res_rdd = rdd.map(lambda x:(x[0],classify_rf(x[1], res, columns)))
print "Accuracy:", 100*(res_rdd.filter(lambda x:x[0] == x[1]).count())/float(res_rdd.count()),"%"
