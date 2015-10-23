import numpy, csv, os
import pickle
import theano

def read_csv_inp(path, filename):
    ifile1  = open(path+filename, "rb")
    ifile2  = open(path+filename, "rb")
    nr_samples = len(list(csv.reader(ifile1)))
    reader = csv.reader(ifile2)
    dat_inp = numpy.zeros([nr_samples,184])
    for idx, row in enumerate(reader):
        dat_inp[idx,:] = row
        #print row
        
    ifile1.close()
    ifile2.close()
    print 'shape of the input ', dat_inp.shape
    print 'is their a NAN ', numpy.any(numpy.isnan(dat_inp))
    return dat_inp
        
def read_csv_tar(path, filename):
    ifile1  = open(path+filename, "rb")
    ifile2  = open(path+filename, "rb")
    nr_samples = len(list(csv.reader(ifile1)))-1
    reader = csv.reader(ifile2)
    dat_tar = numpy.zeros([nr_samples,1    ])
    for idx, row in enumerate(reader):
        if idx > 0:
            colnum = 0
            for col in row:
                if colnum == 1:
                    dat_tar[idx-1,:] = col
                    #print 'target', col
                colnum += 1
                
    ifile1.close()
    ifile2.close()
    print 'shape of the labels ', dat_tar.shape
    print 'is their a NAN ', numpy.any(numpy.isnan(dat_tar))
    return dat_tar


def csv_writer(data, path):
    """
    Write data to a CSV file path
    """
    with open(path, "wb") as csv_file:
        writer = csv.writer(csv_file, delimiter=',')
        for line in data:
            writer.writerow(line)
 
#----------------------------------------------------------------------

 
#host_path = os.getenv('HOME')
host = 'PC'
host = 'HPC'
if host == 'PC':
    path = '/media/aditya/nimi1/repos/data/MICCAI_mri/summary/binary/'
    output_folder = '/media/aditya/nimi1/repos/data/MICCAI_mri/summary/binary/' 
elif host == 'HPC':
    path = '/home/aditya/store/Datasets/MICCAI_mri/summary/binary/'
    output_folder = '/home/aditya/store/Datasets/pickled/' 

dat_inp = read_csv_inp(path,'BinaryTrain_data.csv' )
dat_tar = read_csv_tar(path, 'BinaryTrain_sbj_list.csv')
# tes_inp = read_csv_inp(path,'BinaryTest_data.csv' )
# tes_tar = read_csv_tar(path, 'BinaryTest_sbj_list.csv')

from sklearn import preprocessing
dat_inp_normalized = preprocessing.normalize(dat_inp, norm='l2')

min_max_scaler = preprocessing.MinMaxScaler((-1,1))
dat_inp_minmax = min_max_scaler.fit_transform(dat_inp)
print numpy.amin(dat_inp_minmax,axis=1)
print numpy.amax(dat_inp_minmax,axis=1)

fname = output_folder + 'tarin_normalized.csv'
#csv_writer(dat_inp_normalized, fname)
csv_writer(dat_inp_minmax, fname)


from sklearn.cross_validation import train_test_split

tra_inp_full, val_inp, tra_tar_full, val_tar = train_test_split(dat_inp_minmax, dat_tar, test_size=0.33, random_state=42)
tra_inp, tes_inp, tra_tar, tes_tar = train_test_split(tra_inp_full, tra_tar_full, test_size=0.5, random_state=42)


#tra_inp_full, val_inp, tra_tar_full, val_tar = train_test_split(dat_inp_normalized, dat_tar, test_size=0.33, random_state=42)
#tra_inp, tes_inp, tra_tar, tes_tar = train_test_split(tra_inp_full, tra_tar_full, test_size=0.5, random_state=42)

# tra_inp_full, val_inp, tra_tar_full, val_tar = train_test_split(dat_inp, dat_tar, test_size=0.165, random_state=42)
# tra_inp, tes_inp, tra_tar, tes_tar = train_test_split(tra_inp_full, tra_tar_full, test_size=0.2, random_state=42)

# print
# batch_len = tra_inp.shape[0]
# nr_fold = 20
# nr_samples = batch_len * nr_fold
# rng_seed = 1234
# 
# dat_new_inp = numpy.zeros([nr_samples,184])
# dat_new_tar = numpy.zeros([nr_samples,1  ])
# # numpy.random.seed(1234)
# # bernoulli.rvs
# # tmp = tra_inp * 0.1
# data = tra_inp
# label= tra_tar
# 
# for fold in range(nr_fold):
#     numpy.random.seed(rng_seed+fold)
#     rand_inds = numpy.random.permutation(batch_len) 
#     dat_new_inp[fold*batch_len:(fold+1)*batch_len] = data[rand_inds]
#     dat_new_tar[fold*batch_len:(fold+1)*batch_len] = label[rand_inds]
#     
# tra_inp = numpy.array(dat_new_inp)
# tra_tar = numpy.array(dat_new_tar)


print tra_inp.shape
#print dat_inp[0:2]
print tra_tar.shape
#print dat_tar[0:2]
print val_inp.shape
print val_tar.shape
print tes_inp.shape
print tes_tar.shape
 
targets = set(dat_tar.flatten())
print targets
 
#monitor target balance
for t in targets:
    print 'proportion of target '+str(t)+' in'
    print '    trai set: '+str(numpy.mean(tra_tar==t))
    print '    vali set: '+str(numpy.mean(val_tar==t))
    print '    test set: '+str(numpy.mean(tes_tar==t))
     
print 'tra_tar type ---', type(tra_tar)
      
train_set = []
train_set = (numpy.array(tra_inp).astype(theano.config.floatX),
         numpy.array(tra_tar.flatten()).astype(theano.config.floatX))
print 'nr training instances:  ', len(train_set[0])
print 'nr features:       ',len(train_set[0][0])
print 'nr targets:       ', len(list(set(train_set[1])))
                     
valid_set = []
valid_set = (numpy.array(val_inp).astype(theano.config.floatX),
             numpy.array(val_tar.flatten()).astype(theano.config.floatX))
print 'nr validation instances: ',  len(valid_set[0])
print 'nr features:       ',len(valid_set[0][0])
print 'nr targets:       ', len(list(set(valid_set[1])))
                      
test_set = []
test_set = (numpy.array(tes_inp).astype(theano.config.floatX),
             numpy.array(tes_tar.flatten()).astype(theano.config.floatX))
print 'nr test instances:      ', len(test_set[0])
print 'nr features:       ',len(test_set[0][0])
print 'nr targets:       ', len(list(set(test_set[1])))
              
if not os.path.isdir(output_folder):
    os.makedirs(output_folder)
os.chdir(output_folder) 
      
     
pickle.dump(train_set, open('mri_train.pkl', 'wb'))
pickle.dump(valid_set, open('mri_valid.pkl', 'wb'))
pickle.dump(test_set,  open('mri_test.pkl', 'wb'))
print "done writing"