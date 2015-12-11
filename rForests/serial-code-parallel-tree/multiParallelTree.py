from infoGain import *
from decisionTree import *
from bootstrap import *
import time


def createForest(sc, data, ntrees, column_names, col_num, discrete_column_ids, n_bins=10):
    #sc - pass the SparkContext
    #data - data to be trained on
    #ntrees - number of trees
    #column_names - list of the columns for the data
    #col_num - number of columns to be used for the tree
    #discrete_column_ids - column id's which are continuous and need to be discretized
    #n_bins - number of bins the continuous columns are split into
    #Use in case of memory issues
    if discrete_column_ids == []:
        data = data
    else:
        data = discretize(data, discrete_column_ids, n_bins)

    if ntrees > 5:
        rdd_size = 5
    else:
        rdd_size = ntrees
    n_iterations = int(round(float(ntrees)/rdd_size))

    #Creating the RDD
    forest=[]
    length=len(data)
    # commence bootstrapping
    start_time = time.time()
    for i in range(rdd_size):
        # add tree to forest
        forest.append(sample_wr(data,length))
    print "Time taken to bootstrap data: ", time.time() - start_time
    #Loading the data into a RDD
    forest_rdd=sc.parallelize(forest,4)
    #forest_rdd.cache()
    #Initializing empty result RDD
    #res_rdd = sc.emptyRDD()
    res_list = []
    #creating the RDD union loop
    for i in range(n_iterations):  
        trees = forest_rdd.map(lambda s: create_decision_tree(s, choose_columns(len(column_names),col_num),\
                gain, column_names))
        #trees.cache()
        res_list = res_list + trees.collect()
        #trees.unpersist()

    return res_list,data

def create_Forest(sc, data, ntrees, column_names, num_col, discrete_column_ids=[], n_bins=10):
    #sc - pass the SparkContext
    #data - data to be trained on
    #ntrees - number of trees
    #column_names - list of the columns for the data
    #col_num - number of columns to be used for the tree
    #discrete_column_ids - column id's which are continuous and need to be discretized
    #n_bins - number of bins the continuous columns are split into
    if discrete_column_ids == []:
        data = data
    else:
        data = discretize(data, discrete_column_ids, n_bins)
    #Creating an RDD with bootstrapped data
    forest=[]
    length=len(data)
    # commence bootstrapping
    start_time = time.time()
    for i in range(ntrees):
        # add tree to forest
        forest.append(sample_wr(data,length))
    print "Time taken to bootstrap data: ", time.time() - start_time
    #Loading the data into a RDD
    forest_rdd=sc.parallelize(forest,4)
    #forest_rdd.cache()  
    trees = forest_rdd.map(lambda s: create_decision_tree(s, choose_columns(len(column_names),num_col),\
            gain, column_names)).collect()

    return trees,data
    
    
