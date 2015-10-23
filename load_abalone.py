import numpy, os
import pickle
import csv
import theano

output_folder = '/media/aditya/nimi1/repos/data/abalone/'
ifile  = open('/media/aditya/nimi1/repos/data/abalone/abalone_data.csv', "rb")

# host_path = os.getenv('HOME')
# output_folder=host_path+'/store/Datasets/pickled/'
# ifile  = open('/home/aditya/store/Datasets/abalone/abalone_data.csv', "rb")
reader = csv.reader(ifile)
datas = numpy.zeros((4178,8))
target = numpy.zeros((4178,1))
for idx, row in enumerate(reader):
    for idy, col in enumerate(row):
        if idy == 0:
            dicts = {'M':'0', 'F':'1', 'I':'2'}
            col = dicts[col]
            datas[idx,idy] = col
        elif idy == 8:
            target[idx] = int(col)-1 
        else:
            datas[idx,idy] = col

ifile.close()
#datas = datas.transpose
print datas.shape
#print 

tra_inp = datas[:2500]
print tra_inp.shape
tra_tar = target[:2500]
print tra_tar.shape
val_inp = datas[:633]
val_tar = target[:633]
tes_inp =  datas[3134:]
tes_tar = target[3134:]

targets = set(target.flatten())
print len(targets)
targets = set(tra_tar.flatten())
print len(targets)
targets = set(val_tar.flatten())
print len(targets)
targets = set(tes_tar.flatten())
print len(targets)
# monitor target balance
for t in targets:
    print 'proportion of target '+str(t)+' in'
    print '    trai set: '+str(numpy.mean(tra_tar==t))
    print '    vali set: '+str(numpy.mean(val_tar==t))
    print '    test set: '+str(numpy.mean(tes_tar==t))




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
 
pickle.dump(train_set, open('abalone_train.pkl', 'wb'))
pickle.dump(valid_set, open('abalone_valid.pkl', 'wb'))
pickle.dump(test_set,  open('abalone_test.pkl', 'wb'))
print "done writing"