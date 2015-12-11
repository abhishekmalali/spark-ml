import findspark
findspark.init()
import pyspark
sc = pyspark.SparkContext()
import numpy as np
import time

#Making necessary imports
from decisionTree import *
from infoGain import *
from bootstrap import *
from multiParallelTree import *
import csv
import itertools
data = []
row_num = 0
with open('../data/cov.csv', 'rb') as csvfile:
    spamreader = csv.reader(csvfile, delimiter=',')
    for row in spamreader:
        data.append((row[-1],row[0:-1]))

#Creating column list
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
columns = list(itertools.chain(*names))

#Creating an RDD with bootstrapped data
# n is number of trees in our forest
ntrees=10
start_time = time.time()
res,data_train = create_Forest(sc, data, ntrees, columns, 30, discrete_column_ids=range(10)\
        , n_bins=10) 
print "Time taken to generate the forest: ", time.time() - start_time
#print res

#Checking accuracy
test_rdd = sc.parallelize(data_train)
res_rdd = test_rdd.map(lambda x:(x[0],classify_rf(x[1], res, columns)))
print "Accuracy:", 100*(res_rdd.filter(lambda x:x[0] == x[1]).count())/float(res_rdd.count()),"%"

