#Imports
import autograd.scipy as sci  # Thinly-wrapped scipy
import autograd.numpy as np  # Thinly-wrapped numpy
from scipy.stats import logistic, gumbel_l
from autograd import grad
import os
import pandas as pd
import itertools
import time
import pyspark
import argparse

# MAIN
def main():
    # Parse optional arguments
    parser=argparse.ArgumentParser()
    parser.add_argument("--epochs",help="Number of epochs to iterate over", type=int)
    parser.add_argument("--alpha",help="Step size for gradient descent", type=float)
    parser.add_argument("--func", help="Distribution choice for epsilon. \
                                        Can be norm for normal, log for logistic, \
                                        ,gumbel for gumbel, or r_gumbel for reverse_gumbel")
    args=parser.parse_args()
    # Set default epochs to 1
    if args.epochs:
        epoch = args.epochs
    else:
        epoch = 5

    # Set default alpha to 0.05
    if args.alpha:
        alpha = args.alpha
    else:
        alpha = 0.05

    # Set default distribution to normal
    if args.func in ['log','gumbel','r_gumbel']:
        if args.func == 'log':
            func = log_cdf
        elif args.func == 'gumbel':
            func = gumbel_cdf
        else:
            func = r_gumbel_cdf
    else:
        func = sci.stats.norm.cdf


    #Initialize the Spark Context
    sc=pyspark.SparkContext()
    df=pd.read_csv('../data/musicdata.csv',header=None)
    df.columns=['uid', 'aid', 'rating']

    # I and J are the number of users and artists, respectively
    I = df.uid.max() + 1
    J = df.aid.max() + 1

    # Take the first 2000 samples
    dftouse = df[['rating', 'uid', 'aid']].head(2000)

    # Adjust the indices
    dftouse['uid'] = dftouse['uid'] - 1
    dftouse['aid'] = dftouse['aid'] - 1

    # Take the ratings from 0-100 and transform them from 0-5
    dftouse.rating=np.around((dftouse.rating-1)/20)
    rating_vals = np.arange(1,dftouse.rating.max()+1)
    minR = dftouse.rating.min()
    dftouse['rating'] = dftouse['rating'] - minR
    # R is the number of rating values
    R = len(rating_vals)

    # create buckets as midpoints
    buckets = 0.5 * (rating_vals[1:] + rating_vals[:-1])

    # get length I, J
    I = dftouse.uid.max() + 1
    J = dftouse.aid.max() + 1

    # define some K
    K = 2

    # convert to numpy data matrix
    Xrat = np.array(dftouse)

    # initialize a theta vector with some random values
    theta = np.zeros((I + J) * K + I + J + 1)
    theta = np.random.normal(size=theta.shape[0], loc=0., scale=0.1)

    # define gradll as the gradient of the log likelihood
    gradrll = grad(rowloglikelihood)

    # Open up some files
    f1 = open("out.txt", "w")

    # Now we begin the parallelization!

    # set up parameters
    S=200


    # turn Xrat into an RDD
    xrat_rdd = sc.parallelize(Xrat)
    # Split the ratings into size S chunks
    split_xrat=np.split(Xrat,Xrat.shape[0]/S)
    #And then parallelize those chunks
    split_xrat = sc.parallelize(split_xrat)

    # then run the sgd!

    t=time.time()
    ptheta = split_xrat.map(lambda subX:n_row_sgd(theta, subX, buckets, I, J, K, R, alpha, epoch, gradrll, func)).mean()

    # then we predict (in parallel)
    y_preds = xrat_rdd.map(lambda row: parallel_predict(ptheta, row, buckets, I, J, K)).collect()

    print Xrat[:20,0]
    print y_preds[:20]
    print ptheta[:20]
    # Write things to file
    f1.write("Time (training):  "+ str(time.time()-t)+"\n")
    f1.write("Log likelihood:   "+ str(loglikelihood(ptheta, Xrat, buckets, I, J, K, R, func))+"\n")
    f1.write("Accuracy:         "+ str(accuracy(y_preds, Xrat))+"\n")
    f1.write("RMSE:             "+ str(rmse(y_preds, Xrat)))
    f1.close()

def loglikelihood(theta, Xrat, b, I, J, K, R, func):
    llsum = 0
    # Sum up the log likelihood over the entire dataset, 1 row at at time
    for row in Xrat:
        # get the rating, user index, and artist index
        rating = row[0]
        i = row[1]
        j = row[2]

        # asserts for the indices i, j & rating
        assert i < I and i >= 0
        assert j < J and j >= 0
        assert rating <= R

        # the model for the latent variable
        u_i = theta[K * i:K*(i+1)]
        v_j = theta[(I + j) * K:(I + j + 1) * K]
        a_i = theta[(I + J) * K + i]
        b_j = theta[(I + J) * K + I + j]
        g = theta[(I + J) * K + I + J]
        # beta = theta[(I + J) * K + I + J + 1:]

        # some asserts for the sizes
        assert len(u_i) == K
        assert len(v_j) == K

        # model using latent factors for user/item
        # model = np.dot(u_i, v_j)

        # model using latent factors for user/item and biases
        model = np.dot(u_i, v_j) + a_i + b_j + g

        # edge conditions
        if rating == R:
            llsum += np.log(1 - func(b[rating - 2] + model))
        elif rating == 1:
            llsum += np.log(func(b[rating - 1] + model))
        else:
            llsum += np.log(func(b[rating - 1] + model) - func(b[rating - 2] + model))
    return llsum


## the rowlikelihood for sgd
def rowloglikelihood(theta, row, b, I, J, K, R, func):
    # get the rating, user index, and artist index
    rating = row[0]
    i = row[1]
    j = row[2]

    # asserts for the indices i, j & rating
    assert i < I and i >= 0
    assert j < J and j >= 0
    assert rating <= R

    # the model for the latent variable
    u_i = theta[K * i:K*(i+1)]
    v_j = theta[(I + j) * K:(I + j + 1) * K]
    a_i = theta[(I + J) * K + i]
    b_j = theta[(I + J) * K + I + j]
    g = theta[(I + J) * K + I + J]
    # beta = theta[(I + J) * K + I + J + 1:]

    # some asserts for the sizes
    assert len(u_i) == K
    assert len(v_j) == K

    # model using latent factors for user/item
    # model = np.dot(u_i, v_j)

    # model using latent factors for user/item and biases
    model = np.dot(u_i, v_j) + a_i + b_j + g

    # edge conditions
    if rating == R:
        return np.log(1 - func(b[rating - 2] + model))
    elif rating == 1:
        return np.log(func(b[rating - 1] + model))
    else:
        return np.log(func(b[rating - 1] + model) - func(b[rating - 2] + model))

# Logarithmic cumulative distribution function
def log_cdf(x):
    return 1./(1.+np.exp(-x))

# Gumbel cumulative distribution function
def gumbel_cdf(x):
    return np.exp(-np.exp(-x))

def r_gumbel_cdf(x):
    return gumbel_cdf(-x)

# A function we pass to the array to calculate the updated thetas from each subarray
def n_row_sgd(theta,subX, buckets, I, J, K, R, alpha, epoch, gradrll, func):
    # repeat for each epoch
    for e in range(epoch):
        for row in subX:
            # update theta0 according to current row
            theta += alpha * gradrll(theta, row, buckets, I, J, K, R, func)
    return theta

# given a testsample we derive the class via the latent variable
def parallel_predict(theta, row, buckets, I, J, K):

    rating = row[0]
    i = row[1]
    j = row[2]

     # extract the model for the latent variable
    u_i = theta[K * i:K*(i+1)]
    v_j = theta[(I + j) * K:(I + j + 1) * K]
    a_i = theta[(I + J) * K + i]
    b_j = theta[(I + J) * K + I + j]
    g   = theta[(I + J) * K + I + J]

    # some asserts for the sizes
    assert len(u_i) == K
    assert len(v_j) == K

    model = np.dot(u_i, v_j) + a_i + b_j + g
    Y = -model

    # predict y based on where it lies within the buckets
    return np.sum(Y > buckets) + 1

def accuracy(y_preds, Xrat):
    # get the ratings
    ratings = Xrat[:,0]
    assert len(y_preds) == len(ratings)
    return np.sum(y_preds == ratings)/float(len(y_preds))

def rmse(y_preds, Xrat):
    # get the ratings
    ratings = Xrat[:,0]
    assert len(y_preds) == len(ratings)
    return np.sqrt(np.mean((ratings-y_preds)**2))

if __name__ == "__main__":
    main()
