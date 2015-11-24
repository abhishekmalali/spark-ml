# (c) 2015 by spark-ml Team
# this file will create a folder cache with all data files in it and downloads all files
# that are necessary to run the algorithms


import datetime
from os import rename
from os.path import splitext
from calendar import month_name
from httplib import HTTPConnection
from time import clock
from urllib import urlretrieve
from zipfile import ZipFile
import numpy as np

import os
import errno
import humanize
import sys
import gzip

# file variables
rforest_file_url = 'http://archive.ics.uci.edu/ml/machine-learning-databases/covtype/covtype.data.gz'

# checks if file exists and is readable


def file_exists(path):
    return os.path.isfile(path) and os.access(path, os.R_OK)

# creates directory if it does not exist yet


def create_dir(path):
	if not os.path.exists(path):
	    try:
	        os.makedirs(path)
	    except OSError as error:
	        if error.errno != errno.EEXIST:
	        	raise

# for a given url, this function downloads the file at url to
# cache/filename


def download_file(file_url, cache_path, filename):
    print('downloading %s...' % file_url)

    # show progress in percent & bytes
    def progressHook(num_blocks, block_size, total_size):
        percent = (num_blocks * block_size / (1.0 * total_size)) * 100.0
        sizeH = humanize.naturalsize(num_blocks * block_size, gnu=True)
        total_sizeH = humanize.naturalsize(total_size, gnu=True)
        sys.stdout.write('\r%s / %s \t\t%.2f %%' %
                         (sizeH, total_sizeH, percent))
        sys.stdout.flush()
    # perform download, store result in cache/filename
    print('')
    local_file = urlretrieve(file_url, os.path.join(
        cache_path, filename), progressHook)
    print('')


def main():
	# first create cache directory if it does not exist
	cache_path = os.path.join('.', 'cache')
	create_dir(cache_path)

	# download random forest file
	download_file(rforest_file_url, cache_path, 'covtype.data.gz')

	# decompress file
	print('decompressing file...')
	with gzip.open(os.path.join(cache_path, 'covtype.data.gz'), 'rb') as infile:
	    with open(os.path.join(cache_path, 'covtype.data'), 'w') as outfile:
			for line in infile:
				outfile.write(line)

	# delete gzip
	os.remove(os.path.join(cache_path, 'covtype.data.gz'))
	print('done!')

if __name__ == "__main__":
    main()


