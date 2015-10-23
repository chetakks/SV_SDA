import os
import cPickle
import pickle
import gzip
import os
import sys
import time

import numpy as np
import theano
import theano.tensor as T



def norm(dat_inp):
  
    X_std = (dat_inp - dat_inp.min(axis=0)) / (dat_inp.max(axis=0) - dat_inp.min(axis=0))
    #X_std = (dat_inp - np.min(dat_inp)) / (np.max(dat_inp) - np.min(dat_inp))
    print  
    return X_std

# def norm(dat_inp):
#     from sklearn import preprocessing
#     #print dat_inp
#     #dat_inp_normalized = (dat_inp-np.min(dat_inp))/(np.max(dat_inp)-np.min(dat_inp))
#     #dat_inp_normalized = preprocessing.normalize(dat_inp, norm='l2')
#     #min_max_scaler = preprocessing.MinMaxScaler((-1,1))
#     min_max_scaler = preprocessing.MinMaxScaler((0,1))
#     #min_max_scaler = preprocessing.MinMaxScaler()
#     dat_inp_minmax = min_max_scaler.fit_transform(dat_inp)
#     #dat_inp_minmax = min_max_scaler.fit_transform(dat_inp_normalized)
#     #print np.amin(dat_inp_normalized,axis=1)
#     #print np.amax(dat_inp_normalized,axis=1)
#     #print np.amin(dat_inp_minmax,axis=1)
#     #print np.amax(dat_inp_minmax,axis=1)
#     return dat_inp_minmax
#     #return dat_inp_normalized

def view(data_2view, label):
    import matplotlib.pyplot as plt
    import matplotlib.cm as cm
    plt.Figure
    #for i in range(nr_samp):
    for i in np.where(data_train[:,-1] == label)[0]:       
        #min(len(np.where(data_train[:,-1] == label)[0]),nr_samp)
        
        #plt.imshow(data_train[i,:-1].reshape(150,200), cmap = cm.Greys_r)
        #results_dir = '/media/883E0F323E0F1938/Chetak/Dataset/csip/csip_raw/view_raw/'
        
        
        plt.imshow(data_train[i,:-1].reshape(32,32), cmap = cm.Greys_r)
        plt.colorbar()
        results_dir = '/media/883E0F323E0F1938/Chetak/Dataset/csip/csip_raw/view_gmm/'
        file_name = str(label)+'id'+str(i)+'.png'
        plt.savefig(results_dir+file_name,bbox_inches='tight')
        plt.show()

host_path = os.getenv('HOME')
if host_path == '/home/kinect':
    path = '/media/883E0F323E0F1938/Chetak/Dataset/csip/csip_raw/'
elif host_path == '/home/aditya':
    path = '/home/aditya/store/Datasets/cisp/cisp_raw/'


a = os.listdir(path)
print a

import scipy.io

#data = scipy.io.loadmat(str(path)+'data_normalized.mat')
#data = scipy.io.loadmat(str(path)+'data_nonnormalized.mat')
#data = scipy.io.loadmat(str(path)+'data_sv.mat')
# data = scipy.io.loadmat(str(path)+'data_sv_fixscale.mat')
# data = scipy.io.loadmat(str(path)+'data_sv_full.mat')
data = scipy.io.loadmat(str(path)+'data_sv_sift.mat')
#data = scipy.io.loadmat(str(path)+'data_sv_gist.mat')
#data = scipy.io.loadmat(str(path)+'data_sv_lbp.mat')
#data = scipy.io.loadmat(str(path)+'data_sv_hog.mat')


# data = scipy.io.loadmat(str(path)+'data_sv_fixscale_128.mat')
#data = scipy.io.loadmat(str(path)+'data_sv_fixscale_2.mat')

## Preprocessed data
#data = scipy.io.loadmat(str(path)+'data_dct.mat')
#data = scipy.io.loadmat(str(path)+'data_sv_dct.mat')

# # Domain specialization for training
# settings = ['ar1', 'ar0', 'br0', 'br1', 'bf0', 'cf0', 'cr0', 'cr1', 'df0', 'dr0']
# for idx, setting in enumerate(settings): 
#     print setting
#     data_train_tmp = data['data_'+setting+'_train']
#     if idx == 0:
#         data_train_all = data_train_tmp
#     else:
#         data_train_all = np.vstack((data_train_all, data_train_tmp))
# 
# 
# #data_train = np.vstack((data_train_all, data_train_all))
# data_train = data_train_all
#data_train[:,:-1] = norm(data_train_all[:,:-1])    

def relable_data(dat):
    for idx, val in enumerate(np.unique(dat)):
        dat[dat == val] = idx
    return dat


settings = ['ar1']
settings = ['ar1', 'ar0', 'br0', 'br1', 'bf0', 'cf0', 'cr0', 'cr1', 'df0', 'dr0']

for setting in settings:
    print setting
    data_train = data['data_'+setting+'_train']
    data_valid = data['data_'+setting+'_valid']
    data_test  = data['data_'+setting+'_test']
    
    print
    
    data_train[:,-1] = relable_data(data_train[:,-1])
    data_valid[:,-1] = relable_data(data_valid[:,-1])
    data_test[:,-1]  = relable_data(data_test[:,-1])
    
     
    data_train[:,:-1] = norm(data_train[:,:-1])
    data_valid[:,:-1] = norm(data_valid[:,:-1])
    data_test[:,:-1]  = norm(data_test[:,:-1])

    #view(data_train,label=1)
    #view(data_train,label=2)
    #view(data_train,label=3)
    
    print
      
    train_set = []
    train_set = (np.array(data_train[:,:-1]).astype(theano.config.floatX),
                 np.array(data_train[:,-1]).astype(theano.config.floatX))
    print 'nr training instances:  ', len(train_set[0])
    print 'nr features:      ',len(train_set[0][0])
    print 'nr targets:       ', len(list(set(train_set[1])))
        
    valid_set = []
    valid_set = (np.array(data_valid[:,:-1]).astype(theano.config.floatX),
                 np.array(data_valid[:,-1]).astype(theano.config.floatX))
    print 'nr validation instances: ',  len(valid_set[0])
    print 'nr features:      ',len(valid_set[0][0])
    print 'nr targets:       ', len(list(set(valid_set[1])))
        
    test_set = []
    test_set = (np.array(data_test[:,:-1]).astype(theano.config.floatX),
                np.array(data_test[:,-1]).astype(theano.config.floatX))
    print 'nr test instances: ', len(test_set[0])
    print 'nr features:       ',len(test_set[0][0])
    print 'nr targets:        ', len(list(set(test_set[1])))
      
      
    print
    
    if host_path == '/home/kinect':
        path = '/media/883E0F323E0F1938/Chetak/Dataset/csip/pickled/'
    elif host_path == '/home/aditya':
        path = '/home/aditya/store/Datasets/cisp/pickled/'
    
#     pickle.dump(train_set, open(path+'csip_sv_hog_'+setting+'_train'+'.pkl', 'wb'))
#     pickle.dump(valid_set, open(path+'csip_sv_hog_'+setting+'_valid'+'.pkl', 'wb'))
#     pickle.dump(test_set, open(path+'csip_sv_hog_'+setting+'_test'+'.pkl', 'wb'))
    
#     pickle.dump(train_set, open(path+'csip_sv_lbp_'+setting+'_train'+'.pkl', 'wb'))
#     pickle.dump(valid_set, open(path+'csip_sv_lbp_'+setting+'_valid'+'.pkl', 'wb'))
#     pickle.dump(test_set, open(path+'csip_sv_lbp_'+setting+'_test'+'.pkl', 'wb'))
    
#     pickle.dump(train_set, open(path+'csip_sv_gist_'+setting+'_train'+'.pkl', 'wb'))
#     pickle.dump(valid_set, open(path+'csip_sv_gist_'+setting+'_valid'+'.pkl', 'wb'))
#     pickle.dump(test_set, open(path+'csip_sv_gist_'+setting+'_test'+'.pkl', 'wb'))
#     
    pickle.dump(train_set, open(path+'csip_sift_'+setting+'_train'+'.pkl', 'wb'))
    pickle.dump(valid_set, open(path+'csip_sift_'+setting+'_valid'+'.pkl', 'wb'))
    pickle.dump(test_set, open(path+'csip_sift_'+setting+'_test'+'.pkl', 'wb'))
    
    #pickle.dump(train_set, open(path+'csip_sv22_'+setting+'_train'+'.pkl', 'wb'))
    #pickle.dump(valid_set, open(path+'csip_sv22_'+setting+'_valid'+'.pkl', 'wb'))
    #pickle.dump(test_set, open(path+'csip_sv22_'+setting+'_test'+'.pkl', 'wb'))
    
    #normalized
#     pickle.dump(train_set, open(path+'csip_no'+setting+'_train'+'.pkl', 'wb'))
#     pickle.dump(valid_set, open(path+'csip_no'+setting+'_valid'+'.pkl', 'wb'))
#     pickle.dump(test_set, open(path+'csip_no'+setting+'_test'+'.pkl', 'wb'))
    
    
    #pickle.dump(train_set, open(path+'csip_'+setting+'_train'+'.pkl', 'wb'))
    #pickle.dump(valid_set, open(path+'csip_'+setting+'_valid'+'.pkl', 'wb'))
    #pickle.dump(test_set, open(path+'csip_'+setting+'_test'+'.pkl', 'wb'))
    
    #pickle.dump(train_set, open(path+'csip_ds2_'+setting+'_train'+'.pkl', 'wb'))
    #pickle.dump(valid_set, open(path+'csip_ds2_'+setting+'_valid'+'.pkl', 'wb'))
    #pickle.dump(test_set, open(path+'csip_ds2_'+setting+'_test'+'.pkl', 'wb'))
      
    #pickle.dump(train_set, open(path+'csip_sv_'+setting+'_train'+'.pkl', 'wb'))
    #pickle.dump(valid_set, open(path+'csip_sv_'+setting+'_valid'+'.pkl', 'wb'))
    #pickle.dump(test_set, open(path+'csip_sv_'+setting+'_test'+'.pkl', 'wb'))
    
    #pickle.dump(train_set, open(path+'csip_sv_ds_'+setting+'_train'+'.pkl', 'wb'))
    #pickle.dump(valid_set, open(path+'csip_sv_ds_'+setting+'_valid'+'.pkl', 'wb'))
    #pickle.dump(test_set, open(path+'csip_sv_ds_'+setting+'_test'+'.pkl', 'wb'))
    
    #pickle.dump(train_set, open(path+'csip_sv_l2_'+setting+'_train'+'.pkl', 'wb'))
    #pickle.dump(valid_set, open(path+'csip_sv_l2_'+setting+'_valid'+'.pkl', 'wb'))
    #pickle.dump(test_set, open(path+'csip_sv_l2_'+setting+'_test'+'.pkl', 'wb'))
     
    #pickle.dump(train_set, open(path+'csip_sv_fix_'+setting+'_train'+'.pkl', 'wb'))
    #pickle.dump(valid_set, open(path+'csip_sv_fix_'+setting+'_valid'+'.pkl', 'wb'))
    #pickle.dump(test_set, open(path+'csip_sv_fix_'+setting+'_test'+'.pkl', 'wb'))
     
    #pickle.dump(train_set, open(path+'csip_sv_fix_128_'+setting+'_train'+'.pkl', 'wb'))
    #pickle.dump(valid_set, open(path+'csip_sv_fix_128_'+setting+'_valid'+'.pkl', 'wb'))
    #pickle.dump(test_set, open(path+'csip_sv_fix_128_'+setting+'_test'+'.pkl', 'wb'))
     
    #pickle.dump(train_set, open(path+'csip_sv_fix_2_'+setting+'_train'+'.pkl', 'wb'))
    #pickle.dump(valid_set, open(path+'csip_sv_fix_2_'+setting+'_valid'+'.pkl', 'wb'))
    #pickle.dump(test_set, open(path+'csip_sv_fix_2_'+setting+'_test'+'.pkl', 'wb'))
      
    #pickle.dump(train_set, open(path+'csip_dct_'+setting+'_train'+'.pkl', 'wb'))
    #pickle.dump(valid_set, open(path+'csip_dct_'+setting+'_valid'+'.pkl', 'wb'))
    #pickle.dump(test_set, open(path+'csip_dct_'+setting+'_test'+'.pkl', 'wb'))
      
    #pickle.dump(train_set, open(path+'csip_sv_dct_'+setting+'_train'+'.pkl', 'wb'))
    #pickle.dump(valid_set, open(path+'csip_sv_dct_'+setting+'_valid'+'.pkl', 'wb'))
    #pickle.dump(test_set, open(path+'csip_sv_dct_'+setting+'_test'+'.pkl', 'wb'))
          
    print "done writing"