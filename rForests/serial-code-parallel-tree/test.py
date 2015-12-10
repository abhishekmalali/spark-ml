# Starting spark
#Only use locally


import findspark
findspark.init()
import pyspark
sc = pyspark.SparkContext()
import numpy as np
import time

#Making necessary imports
from decisionTree import *
from infoGain import *

#Function for Bootstrapping data
import random
# method lifted from:http://code.activestate.com/recipes/273085-sample-with-replacement/
def sample_wr(population, k):
    "Chooses k random elements (with replacement) from a population"
    n = len(population)
    _random, _int = random.random, int  # speed hack 
    result = [None] * k
    for i in xrange(k):
        j = _int(_random() * n)
        result[i] = population[j]
    return result
# Choose random columns
def choose_columns(num_cols,m):
    #num_cols is the total number of columns in the dataset
    #m is the number of columns per tree
    return random.sample(range(num_cols),m)


#Code for the balance dataset
# Testing the parallel code
"""
sample = sc.textFile('balance-scale.csv')
rdd=sample.map(lambda x:x.split()).map(lambda x: x[0].strip("'").split(","))\
            .map(lambda x:[v for v in x])\
            .map(lambda x: (str(x[0]),[int(k) for k in x[1:]]))
columns = ['Left-Weight','Left-Distance','Right-Weight','Right-Distance']
print rdd.collect()
#samplec = rdd.collect()
#print samplec

import csv
import itertools
samplec = []
with open('balance-scale.csv', 'rb') as csvfile:
    reader = csv.reader(csvfile, delimiter=',')
    for row in reader:
        samplec.append((str(row[0]),row[1:]))

#print samplec


#Creating an RDD with bootstrapped data
# n is number of trees in our forest
n=10
length=len(samplec)
forest=[samplec]
# commence bootstrapping
start_time = time.time()
for i in range(1,n):
    # add tree to forest
    forest.append(sample_wr(samplec,length))
print "Time taken to bootstrap data: ", time.time() - start_time

#Generating the decision tree parallely
start_time = time.time()
forest_rdd=sc.parallelize(forest)
print forest_rdd.count()

res = forest_rdd.map(lambda s: create_decision_tree(s, choose_columns(len(columns),3), gain, columns)).collect()
print "Time taken to generate the forest: ", time.time() - start_time


#Checking accuracy
res_rdd = rdd.map(lambda x:(x[0],classify_rf(x[1], res, columns)))
print "Accuracy:", 100*(res_rdd.filter(lambda x:x[0] == x[1]).count())/float(res_rdd.count()),"%"
"""

#Code for the forest dataset
# Testing the parallel code
import csv
import itertools
data = []
num_rows = 4000;
row_num = 0
with open('../data/covtype.csv', 'rb') as csvfile:
    spamreader = csv.reader(csvfile, delimiter=',')
    for row in spamreader:
        row_num += 1
        data.append((row[-1],row[0:-1]))
        if row_num > num_rows:
            break

soil_list =[]
for k in range(40):
    string = 'Soil_Type_' + str(k+1)
    soil_list.append(string)
WA_list =[]
for k in range(4):
    string = 'WA_' + str(k+1)
    WA_list.append(string)
names = [['Elevation'], ['Aspect'], ['Slope'], ['HDHyrdo'], ['VDHydro'], ['HDRoadways'], \
         ['9amHills'],['NoonHills'], ['3pmHills'], ['HDFirePoints'], WA_list,\
         soil_list, ['Cover_Type']]
columns_data = list(itertools.chain(*names))

#Discretizing the first 10 columns
data_train = discretize(data,range(10))
#Creating an RDD with bootstrapped data
# n is number of trees in our forest
n=10
length=len(data_train)
#forest=[data_train]
#Checking length
print "Length of the dataset: ", length

forest=[]
# commence bootstrapping
start_time = time.time()
for i in range(n):
    # add tree to forest
    forest.append(sample_wr(data_train,length))
print "Time taken to bootstrap data: ", time.time() - start_time

#Generating the decision tree parallely

forest_rdd=sc.parallelize(forest)
print "Loaded the data"
start_time = time.time()
res = forest_rdd.map(lambda s: create_decision_tree(s, choose_columns(len(columns_data),10), gain, columns_data)).collect()
print "Time taken to generate the forest: ", time.time() - start_time
#print res

#Checking accuracy
test_rdd = sc.parallelize(data_train)
res_rdd = test_rdd.map(lambda x:(x[0],classify_rf(x[1], res, columns_data)))
print "Accuracy:", 100*(res_rdd.filter(lambda x:x[0] == x[1]).count())/float(res_rdd.count()),"%"

