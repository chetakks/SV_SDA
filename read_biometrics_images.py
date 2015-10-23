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
import matplotlib.pyplot as plt



def norm(dat_inp):

    X_std = (dat_inp - dat_inp.min(axis=0)) / (dat_inp.max(axis=0) - dat_inp.min(axis=0))
    print dat_inp
    return X_std

def view(data_2view, label):
    import matplotlib.pyplot as plt
    plt.Figure
    #for i in range(nr_samp):
    for i in np.where(data_train[:,-1] == label)[0]:       
        #min(len(np.where(data_train[:,-1] == label)[0]),nr_samp)
        
        plt.imshow(data_train[i,:-1].reshape(32,32))
        results_dir = '/media/883E0F323E0F1938/Chetak/Dataset/csip/csip_raw/view/'
        file_name = str(label)+'id'+str(i)+'.png'
        plt.savefig(results_dir+file_name,bbox_inches='tight')
        #plt.show()

def split_data_2(dat,split):
# to do: this function could be used in other functions in this file
    """
    Split data into training, validation, and test sets.
    """
    length_all = dat.shape[0]
    length_tra = round(split[0]*length_all)
    length_val = round(split[1]*length_all)
    length_tes = length_all - length_tra - length_val
    tra = dat[0                    :length_tra           ,:] 
    val = dat[length_tra           :length_tra+length_val,:] 
    tes = dat[length_tra+length_val:length_all           ,:]
    print tra.shape, val.shape, tes.shape
    return tra,val,tes

path = '/media/883E0F323E0F1938/Chetak/Dataset/csip/csip_images/'
path = '/media/883E0F323E0F1938/Chetak/Dataset/csip/Cropped/'
#resize_path = '/media/883E0F323E0F1938/Chetak/Dataset/csip/csip_images_resize/'
#resize_path = '/media/883E0F323E0F1938/Chetak/Dataset/csip/csip_images_resize2/'
resize_path = '/media/883E0F323E0F1938/Chetak/Dataset/csip/csip_images_resize3/'
resize_path = '/media/883E0F323E0F1938/Chetak/Dataset/csip/Cropped_resize/'


# adjust width and height to your needs
a = 200#128 #68
b = 128 #68
width = a#32 #150 #200
height = b#32#150

# import Image
# for filename in os.listdir(path):
#     im1 = Image.open(path+filename)
#      
#     # use one of these filter options to resize the image
#     im2 = im1.resize((width, height), Image.NEAREST)      # use nearest neighbour
#     #im3 = im1.resize((width, height), Image.BILINEAR)     # linear interpolation in a 2x2 environment
#     #im4 = im1.resize((width, height), Image.BICUBIC)      # cubic spline interpolation in a 4x4 environment
#     #im5 = im1.resize((width, height), Image.ANTIALIAS)    # best down-sizing filter
#    
#     im2.save(resize_path + filename)
#     #im3.save("BILINEAR" + ext)
#     #im4.save("BICUBIC" + ext)
#     #im5.save("ANTIALIAS" + ext)
    

settings = ['br1']

for setting in settings:
    nr_samples = 0
    print 
    for filename in os.listdir(resize_path):
        sensor = str(filename[5:8])
        if sensor == setting.upper():
            nr_samples = nr_samples +1
    print setting, nr_samples

#nr_samples = 210 #len(np.asanyarray(os.listdir(resize_path)))
#dat_inp = np.zeros([nr_samples,150,200])
#dat_inp = np.zeros([nr_samples,150,150])
#dat_inp = np.zeros([nr_samples,b,a*3])
dat_inp = np.zeros([nr_samples,b,a])
dat_tar = np.zeros([nr_samples,1    ])
sample_ind = 0
print 'total nr_samples', nr_samples
#count = 0
for filename in os.listdir(resize_path):
    id = int(filename[1:4])
    sensor = str(filename[5:8])
    #print sensor, id
    if sensor == 'BR1':
        print sensor, id
        filepath = resize_path+'/'+filename
        import scipy
        #count = count + 1
        #from scipy import misc
        #im = misc.imread(filepath,flatten=True)
        #import Image
        #im = Image.open(filepath).convert('L')
        #im = numpy.array(im)      
        
    ##### Prepare red , blue ,green as three different samples    
    #     tmp  = scipy.ndimage.imread(filepath)
    #     for r in range(3):
    #         im = tmp[:,:,r]
    #         dat_inp[sample_ind,:,:] = im
    #         dat_tar[sample_ind,0] = id
    #         sample_ind += 1
    ## prepare red, blue, green side by side
        #tmp  = scipy.ndimage.imread(filepath)
        #plt.imshow(tmp,cmap=plt.cm.gray);plt.show()
        #im = np.hstack((tmp[:,:,0],tmp[:,:,1],tmp[:,:,2]))
        #plt.imshow(im,cmap=plt.cm.gray);plt.show()
       
    #    
        im = scipy.misc.imread(filepath,flatten=True)
        #im = scipy.ndimage.imread(filepath, mode='L')
        #im = scipy.ndimage.imread(filepath,flatten=True)
#         plt.imshow(im,cmap=plt.cm.gray)
#         plt.title('person: %i' % id)
#         plt.show()
         
        # scale inputs
        print np.max(np.max(im, axis=1))
        im = im / 255
        #plt.imshow(im,cmap=plt.cm.gray)
        #plt.show()
        print im.shape
        print im.size
        print
        #im = im.astype(float)
        dat_inp[sample_ind,:,:] = im
        dat_tar[sample_ind,0] = id
        sample_ind += 1
            
 
nr_samples = len(dat_tar.flatten())
#dat_inp = np.reshape(dat_inp, [nr_samples,150*200])
#dat_inp = np.reshape(dat_inp, [nr_samples,150*150])
#dat_inp = np.reshape(dat_inp, [nr_samples,(a*3)*b])
dat_inp = np.reshape(dat_inp, [nr_samples,a*b])
targets = set(dat_tar.flatten())


# count = 0
# for t in targets:
#     if len(np.where(dat_tar==t)[0]) >= 5:
#         print np.where(dat_tar==t)[0]
#         count = count + len(np.where(dat_tar==t)[0])
#         np.where(dat_tar==t)[0]
#         train_index = np.where(dat_tar==t)[0][:2]
#         valid_index = np.where(dat_tar==t)[0][2:4]
#         test_index  = np.where(dat_tar==t)[0][4:]


def stack(dat_inp,dat_tar,targets,i,j):
    idx = 0 
    for t in targets:
        if len(np.where(dat_tar==t)[0]) >= 5:
            index = np.where(dat_tar==t)[0][i:j]
            if idx == 0:
                inp = dat_inp[index,:]
                tar = dat_tar[index,:]
    
            else:
                inp = np.vstack((inp, dat_inp[index,:]))
                tar = np.vstack((tar, dat_tar[index,:]))
    
            idx = 1
    return inp, tar

# # Domain specialization for training
# settings = ['ar1', 'ar0', 'br0', 'br1', 'bf0', 'cf0', 'cr0', 'cr1', 'df0', 'dr0']
# for idx, setting in enumerate(settings): 
#     print setting
#     data_train_tmp = data['data_'+setting+'_train']
#     if idx == 0:
#         data_train_all = data_train_tmp
#     else:
#         data_train_all = np.vstack((data_train_all, data_train_tmp))
        
        

tra_inp, tra_tar = stack(dat_inp,dat_tar,targets,0,3)     
val_inp, val_tar = stack(dat_inp,dat_tar,targets,3,4)
tes_inp, tes_tar = stack(dat_inp,dat_tar,targets,4,None)  

print np.shape(tra_inp)
print np.shape(tra_tar)   
print np.shape(val_inp)
print np.shape(val_tar)  
print np.shape(tes_inp)
print np.shape(tes_tar)       
print 
        
# #print 'total_nr_samples', count        
# print np.any(np.isnan(dat_inp))
# 
# # shuffle inputs and targets
# np.random.seed(1234)
# rand_inds = np.random.permutation(nr_samples)
# dat_inp = dat_inp[rand_inds,:]
# dat_tar = dat_tar[rand_inds,:]
#  
# split = [0.5,0.25]
# #split = [0.5,0.1]
# # split inputs and targets
# tra_inp,val_inp,tes_inp = split_data_2(dat_inp,split)
# tra_tar,val_tar,tes_tar = split_data_2(dat_tar,split)
#  
# targets = set(dat_tar.flatten())
# print targets       
    



#setting = 'ar1'
settings = ['ar1', 'ar0', 'br0', 'br1', 'bf0', 'cf0', 'cr0', 'cr1', 'df0', 'dr0']


settings = ['br1']
for setting in settings:
    train_set = []
    train_set = (np.array(tra_inp).astype(theano.config.floatX),
             np.array(tra_tar.flatten()).astype(theano.config.floatX))
    print 'nr training instances:  ', len(train_set[0])
    print 'nr features:       ',len(train_set[0][0])
    print 'nr targets:       ', len(list(set(train_set[1])))
                   
    valid_set = []
    valid_set = (np.array(val_inp).astype(theano.config.floatX),
                 np.array(val_tar.flatten()).astype(theano.config.floatX))
    print 'nr validation instances: ',  len(valid_set[0])
    print 'nr features:       ',len(valid_set[0][0])
    print 'nr targets:       ', len(list(set(valid_set[1])))
                    
    test_set = []
    test_set = (np.array(tes_inp).astype(theano.config.floatX),
                 np.array(tes_tar.flatten()).astype(theano.config.floatX))
    print 'nr test instances:      ', len(test_set[0])
    print 'nr features:       ',len(test_set[0][0])
    print 'nr targets:       ', len(list(set(test_set[1])))
    
    print
     
    path = '/media/883E0F323E0F1938/Chetak/Dataset/csip/pickled/'
    pickle.dump(train_set, open(path+'csip_im6_'+setting+'_train'+'.pkl', 'wb'))
    pickle.dump(valid_set, open(path+'csip_im6_'+setting+'_valid'+'.pkl', 'wb'))
    pickle.dump(test_set, open(path+'csip_im6_'+setting+'_test'+'.pkl', 'wb'))
         
    print "done writing"