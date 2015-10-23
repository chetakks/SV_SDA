#!/usr/bin/env python
from numpy import *
  
# array_len = 10000000
#   
# a = zeros(array_len, dtype=float)
#   
# #import pickle
# #f = open('test.pickle', 'wb')
# #pickle.dump(a, f, pickle.HIGHEST_PROTOCOL)
# #f.close()
#   
# import tables
# h5file = tables.openFile('test.h5', mode='w', title="Test Array")
# root = h5file.root
# h5file.createArray(root, "test", a)
# h5file.close()
#  
#  
# import csv
#  
# def csv_writer(data, path):
#     """
#     Write data to a CSV file path
#     """
#     with open(path, "wb") as csv_file:
#         writer = csv.writer(csv_file, delimiter=',')
#         for line in data:
#             writer.writerow(line)

#target_dataset = 'bbbc+moa'
#target_dataset = 'bbbc+comp'
from scipy.sparse import *
from scipy import *
mtx = csr_matrix( (3,4), dtype=int8 )
print mtx.todense()

row = array([0,0,1,2,2,2])
col = array([0,2,2,0,1,2])
data = array([1,2,3,4,5,6])
mtx = csr_matrix( (data,(row,col)), shape=(3,3) )
print mtx.todense()
print mtx.nnz

print

from sklearn.feature_extraction.text import CountVectorizer
vectorizer = CountVectorizer(min_df=1)
print vectorizer 
corpus = [
    'This is the first document.',
    'This is the second second document.',
    'And the third one.',
    'Is this the first document?',
]
X = vectorizer.fit_transform(corpus)
print X                              
print X.todense()
print X.nnz


print




