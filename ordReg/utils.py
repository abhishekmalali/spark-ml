# this file contains several util functions
import pandas as pd
import numpy as np
import itertools

# loads the data from https://archive.ics.uci.edu/ml/datasets/Covertype
# with meaningful columns
def loadCovertypeData(filename):
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

    data = pd.read_table(filename, sep=',',names=col_names, dtype=type_dict)

    return data