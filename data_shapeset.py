import cPickle
import gzip
import os
import sys
import time

import numpy
from numpy import * 
import theano
import theano.tensor as T

import pickle


sys.argv.pop(0)
file_name = sys.argv.pop(0)
print 'file name:', file_name

file_path = '/home/aditya/repos/Database/shapeset/' + file_name
fd = open(file_path + '.amat','r') 
 
datas = []
shapes = []
colors = []
for line in fd:
    if len(line) < ((28*28)+7):
        print line,
    else:
        #print line,
        line_split = line.split()
        datas.append(line_split[0:(28*28)])
        labels = line_split[(28*28):]
        #print labels
        shapes.append(labels[0])
        colors.append(labels[1]) 
fd.close()
#print 'datas:', datas
#print 'shapes', shapes
#print 'colors', colors

dat_set = []
dat_set = (numpy.array(datas).astype(theano.config.floatX),
             numpy.array(shapes).astype(theano.config.floatX))

print 'nr features:       ',len(dat_set[0][0])
print 'nr targets:       ', len(list(set(dat_set[1])))


output_folder= '/home/aditya/repos/Database/shapeset/pickled'
if not os.path.isdir(output_folder):
    os.makedirs(output_folder)
os.chdir(output_folder)  
pickle.dump(dat_set, open(file_name+'.pkl', 'wb'))

    
print "done writing"
#print 'pause \n'
#raw_input()
print 'done'