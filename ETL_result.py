import cPickle
import numpy as np

 
data_path = '/media/aditya/nimi1/repos/Clean_code/results/ETL/'
g1 = open(data_path +  'bbbc_y_test_class.pkl', 'rb')
ground_truth = cPickle.load(g1)
g1.close()
ground_truth = ground_truth.flatten()

nr_reps = 10


En = []
r = []
Ensemble = np.zeros((len(ground_truth),12))
for repetition in range(nr_reps):
    f1 = open(data_path+'bbbc_y_test_class_prob'+str(repetition)+'.pkl', 'rb')
    tmp = cPickle.load(f1)
    f1.close() 
    conv = np.vstack(tmp.flatten()).astype(np.float32)
    En.append(conv)
    print np.shape(En)
    r.append(np.mean(ground_truth == np.array(En[repetition]).argmax(axis=1)))
    print r[repetition]
    Ensemble = np.add(Ensemble,En[repetition])


Avg_error = np.sum(r)/nr_reps
Ensemble_error = np.mean(ground_truth == np.array(Ensemble).argmax(axis=1))
print 'Ensemble_error = ', Ensemble_error
print 'Avg test error = %.2f(%.2f)' % (np.mean(r), np.std(r))  
#'Avg error =', np.mean(r),np.std(r)