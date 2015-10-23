"""
======================================================
Classification of text documents using sparse features
======================================================

This is an example showing how scikit-learn can be used to classify documents
by topics using a bag-of-words approach. This example uses a scipy.sparse
matrix to store the features and demonstrates various classifiers that can
efficiently handle sparse matrices.

The dataset used in this example is the 20 newsgroups dataset. It will be
automatically downloaded, then cached.

The bar plot indicates the accuracy, training time (normalized) and test time
(normalized) of each classifier.

"""

# Author: Peter Prettenhofer <peter.prettenhofer@gmail.com>
#         Olivier Grisel <olivier.grisel@ensta.org>
#         Mathieu Blondel <mathieu@mblondel.org>
#         Lars Buitinck <L.J.Buitinck@uva.nl>
# License: BSD 3 clause

from __future__ import print_function

import logging
import numpy as np
from optparse import OptionParser
import sys
from time import time
import matplotlib.pyplot as plt

from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import HashingVectorizer
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.linear_model import RidgeClassifier
from sklearn.svm import LinearSVC
from sklearn.linear_model import SGDClassifier
from sklearn.linear_model import Perceptron
from sklearn.linear_model import PassiveAggressiveClassifier
from sklearn.naive_bayes import BernoulliNB, MultinomialNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neighbors import NearestCentroid
from sklearn.utils.extmath import density
from sklearn import metrics


# Display progress logs on stdout
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s %(levelname)s %(message)s')


# parse commandline arguments
op = OptionParser()
op.add_option("--report",
              action="store_true", dest="print_report",
              help="Print a detailed classification report.")
op.add_option("--chi2_select",
              action="store", type="int", dest="select_chi2",
              help="Select some number of features using a chi-squared test")
op.add_option("--confusion_matrix",
              action="store_true", dest="print_cm",
              help="Print the confusion matrix.")
op.add_option("--top10",
              action="store_true", dest="print_top10",
              help="Print ten most discriminative terms per class"
                   " for every classifier.")
op.add_option("--all_categories",
              action="store_true", dest="all_categories",
              help="Whether to use all categories or not.")
op.add_option("--use_hashing",
              action="store_true",
              help="Use a hashing vectorizer.")
op.add_option("--n_features",
              action="store", type=int, default=2 ** 16,
              help="n_features when using the hashing vectorizer.")
op.add_option("--filtered",
              action="store_true",
              help="Remove newsgroup information that is easily overfit: "
                   "headers, signatures, and quoting.")

(opts, args) = op.parse_args()
if len(args) > 0:
    op.error("this script takes no arguments.")
    sys.exit(1)

print(__doc__)
op.print_help()
print()


###############################################################################
# Load some categories from the training set
if opts.all_categories:
    categories = None
else:
    categories = [
        'alt.atheism',
        'talk.religion.misc',
        'comp.graphics',
        'sci.space',
    ]

if opts.filtered:
    remove = ('headers', 'footers', 'quotes')
else:
    remove = ()

print("Loading 20 newsgroups dataset for categories:")
print(categories if categories else "all")

data_train = fetch_20newsgroups(subset='train', categories=categories,
                                shuffle=True, random_state=42,
                                remove=remove)

data_test = fetch_20newsgroups(subset='test', categories=categories,
                               shuffle=True, random_state=42,
                               remove=remove)
print('data loaded')

categories = data_train.target_names    # for case categories == None


def size_mb(docs):
    return sum(len(s.encode('utf-8')) for s in docs) / 1e6

data_train_size_mb = size_mb(data_train.data)
data_test_size_mb = size_mb(data_test.data)

print("%d documents - %0.3fMB (training set)" % (
    len(data_train.data), data_train_size_mb))
print("%d documents - %0.3fMB (test set)" % (
    len(data_test.data), data_test_size_mb))
print("%d categories" % len(categories))
print()

# split a training set and a test set
y_train, y_test = data_train.target, data_test.target

print("Extracting features from the training dataset using a sparse vectorizer")
t0 = time()
if opts.use_hashing:
    vectorizer = HashingVectorizer(stop_words='english', non_negative=True,
                                   n_features=opts.n_features)
    X_train = vectorizer.transform(data_train.data)
else:
    vectorizer = TfidfVectorizer(sublinear_tf=True, max_df=0.5,
                                 stop_words='english')
    X_train = vectorizer.fit_transform(data_train.data)
duration = time() - t0
print("done in %fs at %0.3fMB/s" % (duration, data_train_size_mb / duration))
print("n_samples: %d, n_features: %d" % X_train.shape)
print()





print("Extracting features from the test dataset using the same vectorizer")
t0 = time()
X_test = vectorizer.transform(data_test.data)
duration = time() - t0
print("done in %fs at %0.3fMB/s" % (duration, data_test_size_mb / duration))
print("n_samples: %d, n_features: %d" % X_test.shape)
print()

if opts.select_chi2:
    print("Extracting %d best features by a chi-squared test" %
          opts.select_chi2)
    t0 = time()
    ch2 = SelectKBest(chi2, k=opts.select_chi2)
    X_train = ch2.fit_transform(X_train, y_train)
    X_test = ch2.transform(X_test)
    print("done in %fs" % (time() - t0))
    print()


# mapping from integer feature name to original token string
if opts.use_hashing:
    feature_names = None
else:
    feature_names = np.asarray(vectorizer.get_feature_names())
    



targets = set(y_train.flatten())
print(targets)       





import numpy
import os
import theano
import pickle


print("Split the training set into train and valid set")
print()

tmp_X_train = X_train
tmp_y_train = y_train

X_train = tmp_X_train[0:1534]
y_train = tmp_y_train[0:1534]

X_valid = tmp_X_train[1534:]
y_valid = tmp_y_train[1534:]

print("X_train:", X_train.shape)
print("X_valid:", X_valid.shape)
print()


tra_inp = X_train.toarray()
val_inp = X_valid.toarray()
tes_inp = X_test.toarray()

tra_tar = y_train
val_tar = y_valid
tes_tar = y_test

# monitor target balance
for t in targets:
    print('proportion of target '+str(t)+' in')
    print('    trai set: '+str(numpy.mean(tra_tar==t)))
    print('    vali set: '+str(numpy.mean(val_tar==t)))
    print('    test set: '+str(numpy.mean(tes_tar==t)))

#nr_cols = 6
#nr_rows = 1
#show_samples(dat_inp,tra_tar,nr_cols,nr_rows)
print('done')

print('tra_tar type ---', type(tra_tar))
print(numpy.any(numpy.isnan(tra_inp)))

train_set = []
train_set = (numpy.array(tra_inp).astype(theano.config.floatX),
         numpy.array(tra_tar.flatten()).astype(theano.config.floatX))
print('nr training instances:  ', len(train_set[0]))
print('nr features:       ',len(train_set[0][0]))
print('nr targets:       ', len(list(set(train_set[1]))))
               
valid_set = []
valid_set = (numpy.array(val_inp).astype(theano.config.floatX),
             numpy.array(val_tar.flatten()).astype(theano.config.floatX))
print('nr validation instances: ',  len(valid_set[0]))
print('nr features:       ',len(valid_set[0][0]))
print('nr targets:       ', len(list(set(valid_set[1]))))
                
test_set = []
test_set = (numpy.array(tes_inp).astype(theano.config.floatX),
             numpy.array(tes_tar.flatten()).astype(theano.config.floatX))
print('nr test instances:      ', len(test_set[0]))
print('nr features:       ',len(test_set[0][0]))
print('nr targets:       ', len(list(set(test_set[1]))))
         

host_path = os.getenv('HOME')
output_folder=host_path+'/store/Datasets/pickled/20news/'
if not os.path.isdir(output_folder):
    os.makedirs(output_folder)
os.chdir(output_folder) 

def save_batch(dat_set, dat_name):
#import numpy
    batch = 500
    nr_samples = len(dat_set[0])
    nr_batch = int(numpy.floor(nr_samples / batch))
    
    for bat in range(nr_batch):
        tmp = [dat_set[0][bat*batch:(bat+1)*batch],dat_set[1][bat*batch:(bat+1)*batch]]
        pickle.dump(tmp, open(dat_name+'_bat'+str(bat)+'.pkl', 'wb'))
        #pickle.dump(tmp, open('mnist_64x80_'+str(dat_name)+'_bat'+str(bat)+'.pkl', 'wb'))
        print('saving batch num ', bat)
    print('done')


save_batch(train_set,'20news_4_train')
save_batch(valid_set,'20news_4_valid')
save_batch(test_set,'20news_4_test')

# pickle.dump(train_set, open('20news_4_train.pkl', 'wb'))
# pickle.dump(valid_set, open('20news_4_valid.pkl', 'wb'))
# pickle.dump(test_set,  open('20news_4_test.pkl', 'wb'))
 
print("done writing")


