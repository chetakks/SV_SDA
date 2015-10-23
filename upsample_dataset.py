import matplotlib.pyplot as plt
import numpy, scipy
from scipy import interpolate
import os
from load_dataset2 import unpack_data, proportion_of_classes

def resample(kernelIn,outKSize):

    inKSize = len(kernelIn)   
    kernelOut = numpy.zeros((outKSize),numpy.uint8)
    
    x = numpy.arange(inKSize)
    y = numpy.arange(inKSize)
    
    z = kernelIn
    
    xx = numpy.linspace(x.min(),x.max(),outKSize)
    yy = numpy.linspace(y.min(),y.max(),outKSize)
    
    newKernel = interpolate.RectBivariateSpline(x,y,z, kx=2,ky=2)
    
    kernelOut = newKernel(xx,yy)
    return kernelOut


def upsample_data(dat_set):
    import theano
    sample_ind = 0
    nr_samples = len(dat_set[0])
    dat_inp = numpy.zeros([nr_samples,64,80])
    for sample in range(nr_samples):
        im = dat_set[0][sample].reshape(28,28)
        outKSize = 64
        window = resample(im,outKSize)
        offset = 8
        dat_inp[sample_ind,:,offset:64+offset] = window
        #plt.imshow(dat_inp[sample_ind],cmap=plt.cm.gray)
        #plt.show()
        sample_ind += 1
    dat_inp = numpy.reshape(dat_inp, [nr_samples,64*80])
    dat_inp = numpy.array(dat_inp).astype(theano.config.floatX)
    return dat_inp
    


host_path = os.getenv('HOME')
data_path=host_path+'/store/Datasets/pickled/'
#data_path='/media/aditya/nimi1/pickled/'
print 'Loading data ...'
train_set, valid_set, test_set = unpack_data(data_path,'mnist')
train_set = (upsample_data(train_set),train_set[1])
valid_set = (upsample_data(valid_set),valid_set[1])
test_set  = (upsample_data(test_set),test_set[1])

print 'nr training instances:  ', len(train_set[0])
print 'nr features:       ',len(train_set[0][0])
print 'nr targets:       ', len(list(set(train_set[1])))
print 'nr validation instances: ',  len(valid_set[0])
print 'nr features:       ',len(valid_set[0][0])
print 'nr targets:       ', len(list(set(valid_set[1])))
print 'nr test instances:      ', len(test_set[0])
print 'nr features:       ',len(test_set[0][0])
print 'nr targets:       ', len(list(set(test_set[1])))
print 'done'
proportion_of_classes(train_set, valid_set, test_set)

import pickle
#import cPickle as pickle

host_path = os.getenv('HOME')
output_folder=host_path+'/store/Datasets/pickled/'
#output_folder= '/media/aditya/nimi1/repos/data/BBBC/'
#output_folder= '/media/aditya/nimi1/pickled/'
if not os.path.isdir(output_folder):
    os.makedirs(output_folder)
os.chdir(output_folder) 

from sys import getsizeof
print getsizeof(train_set)

# import tables
# h5file = tables.openFile('mnist_64x80_train[0].h5', mode='w', title="Test Array")
# root = h5file.root
# h5file.createArray(root, "train_set", numpy.array(train_set[0]))
# h5file.close()

def save_batch(dat_set, dat_name):
    batch = 1000
    nr_samples = len(dat_set[0])
    nr_batch = nr_samples / batch
    
    for bat in range(nr_batch):
        tmp = (dat_set[0][bat*batch:(bat+1)*batch],dat_set[1][bat*batch:(bat+1)*batch])
        pickle.dump(tmp, open(dat_name+'_bat'+str(bat)+'.pkl', 'wb'))
        #pickle.dump(tmp, open('mnist_64x80_'+str(dat_name)+'_bat'+str(bat)+'.pkl', 'wb'))
        print 'saving batch num', bat
    print 'done'


save_batch(train_set,'mnist_64x80_train')
save_batch(valid_set,'mnist_64x80_valid')
save_batch(test_set,'mnist_64x80_test')

# save_batch(train_set,'train')
# save_batch(valid_set,'valid')
# save_batch(test_set,'test')
# train_set_0 = (train_set[0][0:batch],train_set[1][0:batch])
# train_set_1 = (train_set[0][batch:2*batch],train_set[1][batch:2*batch])
# train_set_2 = (train_set[0][2*batch:3*batch],train_set[1][2*batch:3*batch])
# train_set_3 = (train_set[0][3*batch:4*batch],train_set[1][3*batch:4*batch])
# train_set_4 = (train_set[0][4*batch:5*batch],train_set[1][4*batch:5*batch])
# 
# pickle.dump(train_set_0, open('mnist_64x80_train[0].pkl', 'wb'))
# pickle.dump(train_set_1, open('mnist_64x80_train[1].pkl', 'wb'))
# pickle.dump(train_set_2, open('mnist_64x80_train[2].pkl', 'wb'))
# pickle.dump(train_set_3, open('mnist_64x80_train[3].pkl', 'wb'))
# pickle.dump(train_set_4, open('mnist_64x80_train[4].pkl', 'wb'))
# pickle.dump(valid_set, open('mnist_64x80_valid.pkl', 'wb'))
# pickle.dump(test_set,  open('mnist_64x80_test.pkl', 'wb'))


# pickle.dump(train_set, open('mnist_64x80_train.pkl', 'wb'))
# pickle.dump(valid_set, open('mnist_64x80_valid.pkl', 'wb'))
# pickle.dump(test_set,  open('mnist_64x80_test.pkl', 'wb'))

print "done writing"

