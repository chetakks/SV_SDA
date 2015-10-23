import numpy as np
from sklearn.feature_extraction import image

import csv
    
ifile  = open('/home/aditya/Downloads/week1.csv', "rb")
#ifile  = open('/home/aditya/store/Datasets/bbbc/week1.csv', "rb")
reader = csv.reader(ifile)

rownum = 0
DNA_filename = []
tubulin_filename = []
actin_filename = []

for idx, row in enumerate(reader):
    # Save header row.
    if rownum == 0:
        header = row
    else:
        colnum = 0
        
        for col in row:
            #print '%-8s: %s' % (header[colnum], col)
            if colnum == 2:
                DNA_filename.append(col)
                #print '    DNA_filename', DNA_filename
            elif colnum == 4:
                tubulin_filename.append(col)
                #print '    tubulin_filename', tubulin_filename
            elif colnum == 6:
                actin_filename.append(col)
                #print '    actin_filename', actin_filename
 
            colnum += 1
            
    rownum += 1
    #print rownum
    #if rownum == 5:
    #    break

ifile.close()
#print

import os
import numpy
import scipy
import scipy.ndimage
import matplotlib.pyplot as plt
import random
import theano
import theano.tensor as T
#import pickle
from sys import getsizeof
import cPickle as pickle
#import marshal as pickle
#import cPickle
host_path = os.getenv('HOME')
output_folder=host_path+'/store/Datasets/pickled/'
output_folder= '/media/aditya/nimi1/repos/data/BBBC/' 
 
dir = '/home/aditya/Downloads/Week1_22123/'
#dir = '/home/aditya/store/Datasets/bbbc/Week1_22123/'
sample_ind = 0
nr_samples = len(os.listdir(dir))
print 'nr_samples', nr_samples
nr_samples = 100
print 'nr_samples', nr_samples
#nr_samples = nr_samples * 64
nr_samples = nr_samples * 128
#nr_samples = 25

#dat_inp = numpy.zeros([nr_samples,1024,1280])
#dat_inp = numpy.zeros([nr_samples,256,320])
#dat_inp = numpy.zeros([nr_samples,128,160])
dat_inp = numpy.zeros([nr_samples,64,80])
dat_tar = numpy.zeros([nr_samples,1    ])
print 'Loading the data ...' 
count = 0
count_im = 0
rand_mean_value = []

rmv = numpy.zeros(8)
rl = 0
for filename in os.listdir(dir):
    #print filename
    
    for item in DNA_filename:
        if item == filename:
            target = 0  # DNA
            #print 'DNA_file_found'
    for item in tubulin_filename:
        if item == filename:
            target = 1  # tubulin
            #print 'tubulin_file_found'
    for item in actin_filename:
        if item == filename:
            target = 2  # actin
            #print 'actin_file_found'

    dat_tar[sample_ind,0] = target
    print target
    filepath = dir+'/'+filename
    #import scipy
    #from scipy import misc
    #im = misc.imread(filepath,flatten=True)
    #import Image
    #im = Image.open(filepath).convert('L')
    #im = numpy.array(im)      
    #im = scipy.ndimage.imread(filepath)
    #im = scipy.ndimage.imread(filepath, mode='L')
    im = scipy.ndimage.imread(filepath,flatten=True)
    
    # scale inputs
    #print numpy.max(numpy.max(im, axis=1))
    im = im / 2**16.
    #plt.imshow(im,cmap=plt.cm.gray)
    #plt.show()
    print im.shape
    print im.size
    im = im.astype(float)
    
    # Define the window size
    #windowsize_r = 128
    #windowsize_c = 160
    windowsize_r = 64
    windowsize_c = 80
    test_image = im
    ten_images = numpy.zeros([10,1024,1280])
    
    #for p in range(5):
    if target == 2:
        #count_im += 1
        #ten_images[count,:,:] =test_image
        #count += 1
        #print 'count', count
        #if count == 9:
            #five_images = np.arange(5 * 4 * 4 * 3).reshape(5, 4, 4, 3)
            #patches = image.PatchExtractor((200, 200), max_patches=6,  random_state=0).transform(ten_images)
        patches = image.extract_patches_2d(test_image, (100, 100), max_patches=6,  random_state=0)
        for r in range(6):
            #print 'r', r
            
#             if patches[r].mean() > 0.015 and patches[r].mean() < 0.017:
#                 plt.imshow(patches[r],cmap=plt.cm.gray)
#                 plt.show() 
            
            print 'count_im = ', count_im
            plt.imshow(patches[r],cmap=plt.cm.gray)
            plt.show()
            
            
#             if count_im == 6 or count_im == 8 or count_im == 9 or count_im == 10 or count_im == 14 or count_im == 16 or count_im == 37 or count_im == 38:
#                 print 'random avg = ', patches[r].mean()
#                 rmv[rl] = patches[r].mean()
#                 rl +=1
#                 print 'rand_mean_value', rmv
#                 print 'rand_mean_value', rmv[0:rl]
#                 print 'Avg rand_mean_value', numpy.mean(rmv[0:rl])
                
                
            count_im += 1
            print 'all avg = ', patches[r].mean()
    print
        
#         # Crop out images into windows
#         count = 0
#         for r in range(0,test_image.shape[0]+1 - windowsize_r, windowsize_r):
#             for c in range(0,test_image.shape[1]+1 - windowsize_c, windowsize_c):
#                 window = test_image[r:r+windowsize_r,c:c+windowsize_c]
#                 dat_inp[sample_ind,:,:] = window
#                 dat_tar[sample_ind,0] = target
#                 
#                 #print target
#                 plt.imshow(window,cmap=plt.cm.gray)
#                 plt.show()
#                 #print window
#                 #print numpy.max(numpy.max(window, axis=1))
#                 sample_ind += 1
#                 count += 1
        #print count

    
    # subsample
    #im = scipy.misc.imresize(im, [256,320], interp='bicubic' )
    #im = scipy.misc.imresize(im, [64,80], interp='bicubic' )
    #plt.imshow(im,cmap=plt.cm.gray)
    #plt.show()
    
    #im = scipy.ndimage.filters.gaussian_filter(im,1.14)
    #dat_inp[sample_ind,:,:] = im

    #sample_ind += 1
    if sample_ind == 100 * 128:  #100*64:
        break
#print dat_tar

# reshape each 2d image as one row
print dat_inp.shape

# one_image = np.arange(4 * 4 * 3).reshape((4, 4, 3))
# print one_image[:, :, 0]  # R channel of a fake RGB picture
# # array([[ 0,  3,  6,  9],
# #        [12, 15, 18, 21],
# #        [24, 27, 30, 33],
# #        [36, 39, 42, 45]])
# 
# patches = image.extract_patches_2d(one_image, (2, 2), max_patches=2,  random_state=0)
# print patches.shape
# # (2, 2, 2, 3)
# print patches[:, :, :, 0]
# # array([[[ 0,  3],
# #         [12, 15]],
# # 
# #        [[15, 18],
# #         [27, 30]]])
# patches = image.extract_patches_2d(one_image, (2, 2))
# print patches.shape
# # (9, 2, 2, 3)
# print patches[:, :, :, 0]
# array([[15, 18],
#        [27, 30]])