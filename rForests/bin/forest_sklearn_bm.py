import pandas as pd
import itertools
import numpy as np
from sklearn.cross_validation import train_test_split
from sklearn.ensemble import RandomForestClassifier
import time

#For column names and data types to be loaded in correctly
soil_list =[]
for k in range(40):
    string = 'Soil_Type_' + str(k+1)
    soil_list.append(string)
WA_list =[]
for k in range(4):
    string = 'WA_' + str(k+1)
    WA_list.append(string)
names = [['Elevation'], ['Aspect'], ['Slope'], ['HDHyrdo'], \
        ['VDHydro'], ['HDRoadways'], \
         ['9amHills'],['NoonHills'], ['3pmHills'], ['HDFirePoints'], WA_list,\
         soil_list, ['Cover_Type']]

type_list = []
for k in range(55):
    if k<6:
        type_list.append(float)
    elif k == 9:
        type_list.append(float)
    else:
        type_list.append(int)

col_names = list(itertools.chain(*names))
type_dict = {}
assert(len(col_names) == len(type_list))
for k in range(len(col_names)):
    type_dict[col_names[k]] = type_list[k]

#Reading in data
data = pd.read_table('../data/covtype.data',sep=',',names=col_names,\
        dtype=type_dict)

#Splitting into attributes for training and response variables
X = data[col_names[:-1]]
Y = data[col_names[-1]]
#Splitting into test and training sets
itrain, itest = train_test_split(xrange(X.shape[0]), train_size=0.7)
mask=np.ones(X.shape[0], dtype='int')
mask[itrain]=1
mask[itest]=0
mask = (mask==1)
X_train = X[mask]
X_test = X[~mask]
Y_train = Y[mask]
Y_test = Y[~mask]
#Implementing RandomForests in SKLearn
clf = RandomForestClassifier(n_estimators=500)
start_time = time.time()
clf.fit(X_train,Y_train)
print 'Training time for 500 estimator trees:',time.time()-start_time
Y_pred = clf.predict(X_test)
accuracy = float(sum(Y_pred-Y_test == 0))/len(Y_test)
print '%accuracy for Random Forests:', accuracy*100


