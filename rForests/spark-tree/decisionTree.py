import copy
from infoGain import *

def choose_attribute(sampled_rdd,attributes, count):
    #sampled_rdd: contains the data in a labeled point format
    #attributes: list of columns - pass as indices
    #count: count of the RDD to avoid repeated count calculations which are expensive.

    best_gain = float("inf")
    best_attr = None
    for att in attributes:
        gain = infoGain2(sampled_rdd,count,att)
        #print att,gain
        if gain <= best_gain:
            best_gain = gain
            best_attr = att
    if best_attr == None:
        return attributes[0], sampled_rdd.map(lambda x:x[1][attributes[0]]).distinct().collect()
    cats=sampled_rdd.map(lambda x:x[1][best_attr]).distinct().collect()
    return best_attr,cats

#Function returns most frequent value in response variable
def most_frequent(data):
    highest_freq = 0
    most_freq = None
    vals = data.map(lambda x:x[0]).distinct().collect()
    for val in vals:
        freq = data.filter(lambda x:x[0] == val)\
                    .map(lambda x:x[0]).count()
        if freq > highest_freq:
            most_freq = val
            highest_freq = freq
    return most_freq



#All trees are referenced on the column names instead of the indices since it improves readability.

#Creates a decision tree as a nested dictionary
def createDecisionTree(sub_rdd,attributes,columns):
    #sub_rdd : RDD which has the data [Might be filtered since the tree process is recursive]
    #attributes: Column numbers to be used for building the tree
    #columns: list constaining the names of the columns
    if len(attributes) <= 0:
        return most_frequent(sub_rdd)
    elif sub_rdd.map(lambda x:x[0]).distinct().count() == 1:
        return sub_rdd.map(lambda x:x[0]).distinct().collect()[0][0]
    elif sub_rdd.count() == 0:
        return 0
    else:
        bestAttr,vals = choose_attribute(sub_rdd,attributes,sub_rdd.count())
        attributes.remove(bestAttr)
        #print bestAttr,vals
        tree = {columns[bestAttr]:{}}
        for val in vals:
            new_rdd = sub_rdd.filter(lambda x:x[1][bestAttr] == val)\
            .map(lambda x:(x[0],x[1]))
            #print val,bestAttr,attributes
            new_attributes = copy.deepcopy(attributes)
            subtree = createDecisionTree(new_rdd,new_attributes,columns)
            tree[columns[bestAttr]][val] = subtree
    return tree

#Function for classifying the data
def classify_tree(data,columns,tree):
    #data: second half of the tuple in the RDD based on the labeledPoint format
    #columns: contains the name of columns for referencing the tree
    #tree: Pass the tree object which has been trained
    if type(tree) == type("string"):
        return tree
    else:
        attr = tree.keys()[0]
        t = tree[attr][data[columns.index(attr)]]
        return classify_tree(data,columns,t)
