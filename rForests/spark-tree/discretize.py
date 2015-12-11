#This file ahs the code to discretize the data in continuous columns
import operator
from bisect import bisect
import numpy as np
def discrete_val(data,ranges):
    k = bisect(ranges,float(data)) 
    if k == 0:
        return str(ranges[k])+"<"
    elif k == len(ranges):
        return str(ranges[k-1])+">"
    else:
        return str(ranges[k-1])+ "-" + str(ranges[k])


def discretize_column(data,column,n_bins):
    col_max = data.map(lambda x:float(x[1][column])).max()
    col_min = data.map(lambda x:float(x[1][column])).min()
    ranges = list(np.linspace(col_min, col_max, n_bins))
    new_data = data.map(lambda x:(x[0],[x[1][0:column],[discrete_val(x[1][column],ranges)],x[1][column+1:]]))
    new_data = new_data.map(lambda x:(x[0],reduce(operator.add, x[1])))
    data = new_data
    return data

def discretize_columns(data,column_ids,n_bins=10):
    col_count = 0
    for column in column_ids:
        if col_count == 0:
            return_rdd = discretize_column(data,column,n_bins)
        else:
            return_rdd = discretize_column(return_rdd,column,n_bins)
        col_count += 1
    return return_rdd
