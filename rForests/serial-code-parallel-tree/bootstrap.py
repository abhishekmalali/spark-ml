#Function for Bootstrapping data
import random
# method lifted from:http://code.activestate.com/recipes/273085-sample-with-replacement/
def sample_wr(population, k):
    "Chooses k random elements (with replacement) from a population"
    n = len(population)
    _random, _int = random.random, int  # speed hack 
    result = [None] * k
    for i in xrange(k):
        j = _int(_random() * n)
        result[i] = population[j]
    return result
# Choose random columns
def choose_columns(num_cols,m):
    #num_cols is the total number of columns in the dataset
    #m is the number of columns per tree
    return random.sample(range(num_cols),m)
