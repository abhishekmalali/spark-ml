from collections import Counter
from collections import defaultdict
import numpy as np
"""
This module holds functions that are responsible for creating a new
decision tree and for using the tree for data classificiation.
"""

def majority_value(data):
    """
    Creates a list of all values in the target attribute for each record
    in the data list object, and returns the value that appears in this list
    the most frequently.
    """
    return most_frequent([record[0] for record in data])

def most_frequent(lst):
    """
    Returns the item that appears most frequently in the given list.
    """
    lst = lst[:]
    highest_freq = 0
    most_freq = None

    for val in unique(lst):
        if lst.count(val) > highest_freq:
            most_freq = val
            highest_freq = lst.count(val)
            
    return most_freq

def unique(lst):
    """
    Returns a list made up of the unique values found in lst.  i.e., it
    removes the redundant values in lst.
    """
    lst = lst[:]
    unique_lst = []

    # Cycle through the list and add each value to the unique list only once.
    for item in lst:
        if unique_lst.count(item) <= 0:
            unique_lst.append(item)
            
    # Return the list with all redundant values removed.
    return unique_lst

def get_values(data, attr):
    """
    Creates a list of values in the chosen attribute for each record in data,
    prunes out all of the redundant values, and return the list.  
    """
    try:
        return unique([record[1][attr] for record in data])
    except:
        return data

def choose_attribute(data, attributes, fitness):
    """
    Cycles through all the attributes and returns the attribute with the
    highest information gain (or lowest entropy).
    """
    best_gain = 0.0
    best_attr = None

    for attr in attributes:
        gain = fitness(data, attr)
        if gain >= best_gain:
            best_gain = gain
            best_attr = attr
                
    return best_attr

def get_examples(data, attr, value):
    """
    Returns a list of all the records in <data> with the value of <attr>
    matching the given value.
    """
    rtn_lst = []
    if not data:
        return rtn_lst
    else:
        for record in data:
            try:
                if record[1][attr]==value:
                    rtn_lst.append(record)
            except:
                continue
    return rtn_lst

def create_decision_tree(data, attributes, fitness_func, columns):
    """
    Returns a new decision tree based on the examples given.
    """
    vals = [x[0] for x in data]
    default = most_frequent(vals)

    # If the dataset is empty or the attributes list is empty, return the
    # default value. When checking the attributes list for emptiness, we
    # need to subtract 1 to account for the target attribute.
    if not data or (len(attributes) - 1) <= 0:
        return default
    # If all the records in the dataset have the same classification,
    # return that classification.
    elif vals.count(vals[0]) == len(vals):
        return vals[0]
    else:
        # Choose the next best attribute to best classify our data
        best = choose_attribute(data, attributes,
                                fitness_func)

        # Create a new decision tree/node with the best attribute and an empty
        # dictionary object--we'll fill that up next.
        tree = {columns[best]:{}}

        # Create a new decision tree/sub-node for each of the values in the
        # best attribute field
        for val in get_values(data, best):
            # Create a subtree for the current value under the "best" field
            subtree = create_decision_tree(
                get_examples(data, best, val),
                [attr for attr in attributes if attr != best],
                fitness_func, columns)

            # Add the new subtree to the empty dictionary object in our new
            # tree/node we just created.
            try:
                tree[columns[best]][val] = subtree
            except:
                continue

    return tree

# Function to classify
def get_classification(record, tree, columns):
    """
    This function recursively traverses the decision tree and returns a
    classification for the given record.
    """
    # If the current node is a string, then we've reached a leaf node and
    # we can return it as our answer
    if type(tree) == type("string"):
        return tree

    # Traverse the tree further until a leaf node is found.
    else:
        attr = tree.keys()[0]
        t = tree[attr][record[columns.index(attr)]]
        return get_classification(record, t, columns)

# Function to choose the best value in the forest.
def classify_rf(data, tree_list, columns):
    res = []
    for tree in tree_list:
        try:
            res.append(get_classification(data,tree,columns))
        except:
            continue
    
    result = Counter(res)
    return result.most_common(1)[0][0]
    """  
    (values,counts) = np.unique(res,return_counts=True)
    ind=np.argmax(counts)
    return values[ind]
    """
