from multiParallelTree import *
from infoGain import *
import csv
import itertools
import time

#Importing spark
import findspark
findspark.init()
import pyspark
sc = pyspark.SparkContext()

#Loading the data
data = []
num_rows = 30000;
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

#Creating the tree
n_trees = 50;
start_time = time.time()
res_trees = createForest(sc, data, n_trees, columns_data, range(10),n_bins=10)
print "Time taken to train :" ,time.time() - start_time

#Checking the accuracy
data_train = discretize(data,range(10))
test_rdd = sc.parallelize(data_train)
res_rdd = test_rdd.map(lambda x:(x[0],classify_rf(x[1], res_trees, columns_data)))
print "Accuracy:", 100*(res_rdd.filter(lambda x:x[0] == x[1]).count())/float(res_rdd.count()),"%"

