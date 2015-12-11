# CS 205 Final Project

This is the official repo for the CS205 final project *Implementation of Spark Machine learning algorithms* to model categorical data. A detailed documentation can be found at <http://abhishekmalali.github.io/spark-ml/>

## Ordinal Regression
For running the ordinal regression code, a small dataset (10K entries) has been supplied. A tutorial documentation is furthermore provided giving an in-depth deviation of the idea and motivation behind ordinal regression.

##Random Forests
For running the random forest code, three datasets have been provided with the github repository. We have also written three custom scripts for each of the dataset. Since the datasets have continuous variables which we have to explicitly define in the function, we wrote the scripts with the required parameters. Within the rForests folder, we have two directories. The first directory serial-code-parallel-tree implements the first method of parallelization as described on our website. The test files are provided along with all the base code for this implementation. In case the code is being run on AWS include --executor-memory command with spark submit since the implementation is memory intensive.

For the second implementation, we were able to create trees using only actions and transformations. The test cases have been written for the datasets along with the base code for the implementation. This method is slow and cumbersome. All files are present in the spark-tree directory.

---
(c) 2015 Chainani, Malali, Spiegelberg
