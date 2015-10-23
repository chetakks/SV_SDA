# import pickle
# import cPickle
# data_path = '/home/aditya/store/Datasets/pickled/bbbc+feat/'
# f1 = open(data_path+'bbbc+feat_valid.pkl', 'rb')
# valid_set = cPickle.load(f1)
# print 'done'
# print 'nr validation instances: ',  len(valid_set[0])
# print 'nr features:       ',len(valid_set[0][0])
# print 'nr targets:       ', len(list(set(valid_set[1])))
# f2 = open(data_path+'bbbc+feat_test.pkl', 'rb')
# test_set = cPickle.load(f2)
# print 'nr test instances:      ', len(test_set[0])
# print 'nr features:       ',len(test_set[0][0])
# print 'nr targets:       ', len(list(set(test_set[1])))


import cPickle
import gzip


def save(object, filename, protocol = -1):
    """Save an object to a compressed disk file.
       Works well with huge objects.
       By Zach Dwiel.
    """
    file = gzip.GzipFile(filename, 'wb')
    cPickle.dump(object, file, protocol)
    file.close()
    
def load(filename):
    """Loads a compressed object from disk.
       By Zach Dwiel.
    """
    file = gzip.GzipFile(filename, 'rb')
    object = cPickle.load(file)
    file.close()

    return object

filename =  '/home/aditya/store/Datasets/pickled/bbbc+feat/bbbc+feat_gzip_train.pkl'
filename =  '/media/883E0F323E0F1938/Chetak/Dataset/pickled/mnist2_train.pkl'
f = open('/media/883E0F323E0F1938/Chetak/Dataset/pickled/mnist2_train.pkl', 'rb')
train_set = cPickle.load(f)
f.close()

#train_set = load(filename)
print 'nr training instances:  ', len(train_set[0])
print 'nr features:       ',len(train_set[0][0])
print 'nr targets:       ', len(list(set(train_set[1])))
print
