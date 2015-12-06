import math
import copy
from bisect import bisect
import numpy as np

def entropy(data):
    """
    Calculates the entropy of the given data set for the target attribute.
    """
    val_freq = {}
    data_entropy = 0.0

    # Calculate the frequency of each of the values in the target attr
    for record in data:
        if (val_freq.has_key(record[0])):
            val_freq[record[0]] += 1.0
        else:
            val_freq[record[0]] = 1.0

    # Calculate the entropy of the data for the target attribute
    for freq in val_freq.values():
        data_entropy += (-freq/len(data)) * math.log(freq/len(data), 2) 
        
    return data_entropy
    
def gain(data, attr):
    """
    Calculates the information gain (reduction in entropy) that would
    result by splitting the data on the chosen attribute (attr).
    """
    val_freq = {}
    subset_entropy = 0.0

    # Calculate the frequency of each of the values in the target attribute
    for record in data:
        if (val_freq.has_key(record[1][attr])):
            val_freq[record[1][attr]] += 1.0
        else:
            val_freq[record[1][attr]] = 1.0

    # Calculate the sum of the entropy for each subset of records weighted
    # by their probability of occuring in the training set.
    for val in val_freq.keys():
        val_prob = val_freq[val] / sum(val_freq.values())
        data_subset = [record for record in data if record[1][attr] == val]
        subset_entropy += val_prob * entropy(data_subset)  
    # Subtract the entropy of the chosen attribute from the entropy of the
    # whole data set with respect to the target attribute (and return it)
    return (entropy(data) - subset_entropy)

#Function for discretizing the columns before creating the decision tree
def discrete_val(data,ranges):
    k = bisect(ranges,float(data))
    if k == 0:
        return str(ranges[k])+"<"
    elif k == len(ranges):
        return str(ranges[k-1])+">"
    else:
        return str(ranges[k-1])+ "-" + str(ranges[k])


def discretize(data,column_ids,n_bins=10):
    new_data = copy.deepcopy(data)
    for column in column_ids:
        k = [float(row[1][column]) for row in new_data]
        col_max = max(k);
        col_min = min(k);
        ranges = list(np.linspace(col_min, col_max, n_bins))
        for idx in range(len(data)):
            new_data[idx][1][column] = discrete_val(data[idx][1][column],ranges)
    return new_data
