from collections import Counter
import numpy as np

#Calculates Information gain value
def IG(l):
    length=len(l)
    c=Counter()
    for v in l:
        c[v] += 1.0/length
    return 1-sum(np.multiply(c.values(),c.values()))

#Calculates the aggregated Information gain for an attribute
def infoGain2(sampled_rdd,count,attr=0):
    sampled_rdd.cache()
    try:
        print output.collect()
        output = sampled_rdd.map(lambda x: (x[1][attr],x[0]))\
            .groupByKey().mapValues(lambda x: tuple(x))\
            .map(lambda x: (x[0],x[1], len(x[1])/float(count)))\
            .map(lambda x: (x[0],IG(x[1]),x[2]))\
            .map(lambda x: x[1]*x[2]).reduce(lambda a,b:a+b)
    except:
        output = sampled_rdd
    return output
