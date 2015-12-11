#Importing spark
import findspark
findspark.init()
import pyspark
sc = pyspark.SparkContext()
import time
#Importing the decisionTree module
import csv
from decisionTree import *
from infoGain import *
from bootstrap import *
from multiParallelTree import *
##Loading dataset with continuous features
sample = []
with open('../data/crx.csv', 'rb') as csvfile:
    reader = csv.reader(csvfile, delimiter=',')
    for row in reader:
        sample.append((str(row[-1]),row[0:-1]))
columns = []
for k in range(1,16):
    columns.append('A'+str(k))

#Training the decision tree
ntrees = 500
start_time = time.time()
tree, data = create_Forest(sc, sample, ntrees, columns, 10, discrete_column_ids=[1,2,7,10,13,14], n_bins=10)
print 'Time taken to train tree:',time.time() - start_time


#Testing the accuracy of the decision tree
rdd = sc.parallelize(data,4)
res_rdd = rdd.map(lambda x:(x[0],classify_rf(x[1], tree, columns)))
#print res_rdd.collect()
print "Accuracy:", 100*(res_rdd.filter(lambda x:x[0] == x[1]).count())/float(res_rdd.count()),'%'


