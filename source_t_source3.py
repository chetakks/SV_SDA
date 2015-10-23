import cPickle
import pickle
import os
import sys
import time
import csv 

import numpy
from numpy import *



def load_outputs(dir, prefix='outputs_'):
    outputs_list = []
    for filename in os.listdir(dir): 
        if filename.startswith(prefix):
            outputs = pickle.load(open(dir+'/'+filename, 'rb'))
            outputs_list.append(outputs)
    return outputs_list

def reuse_SdA6(finetune_lr=None, pretraining_epochs=None,
             pretrain_lr=None, training_epochs=None,
              n_ins=None,
              hidden_layers_sizes=None,
              dataset_A=None, 
              n_outs=None,
              retrain=None,
              source_reuse_mode=None,
              #reset_pt=None,
              dataset_B=None,
              n_outs_source=None,
              batch_size=None,
              output_fold = None, 
              rng_seed=None,
              retrain_ft_layers=None,
              sda_reuse_pt_model=None, 
              sda_reuse_ft_model=None, 
              repetition=None,
              tau=None,
              training_data_fraction=None,
              #dropout = None,
              dropout_rate = None):
    

    """
    Demonstrates how to train and test a stochastic denoising autoencoder.

    This is demonstrated on MNIST.

    :type learning_rate: float
    :param learning_rate: learning rate used in the finetune stage
    (factor for the stochastic gradient)

    :type pretraining_epochs: int
    :param pretraining_epochs: number of epoch to do pretraining

    :type pretrain_lr: float
    :param pretrain_lr: learning rate to be used during pre-training

    :type n_iter: int
    :param n_iter: maximal number of iterations ot run the optimizer

    :type dataset: string
    :param dataset: path the the pickled dataset

    """
    
    # Import sandbox.cuda to bind the specified GPU to this subprocess
    # then import the remaining theano and model modules.
    import theano.sandbox.cuda
    theano.sandbox.cuda.use('gpu0')
    
    import theano
    import theano.tensor as T
    from theano.tensor.shared_randomstreams import RandomStreams
    from load_dataset2 import load_dataset, dropout_weights, combine_two_dataset
    from mlp import HiddenLayer
    from dA5 import dA5
    from SdA6 import SdA6
    from mlp5_train_model2 import train_test_mlp
    
    if source_reuse_mode is not 'Join':
        if dataset_A == 'mnist_64x80':
            datasets = load_dataset(dataset_A,reduce=5,samples=None, frac=training_data_fraction, rng_seed=rng_seed+repetition)
        elif dataset_A == 'bbbc':
            datasets = load_dataset(dataset_A,reduce=5,samples=None, frac=training_data_fraction, rng_seed=rng_seed+repetition)
        elif dataset_A == 'bbbc+feat':
            datasets = load_dataset(dataset_A,reduce=5,samples=None, frac=training_data_fraction, rng_seed=rng_seed+repetition)
        elif dataset_A == 'bbbc+feat2':
            datasets = load_dataset(dataset_A,reduce=5,samples=None, frac=training_data_fraction, rng_seed=rng_seed+repetition)
        elif dataset_A == 'bbbc+feat3':
            datasets = load_dataset(dataset_A,reduce=5,samples=None, frac=training_data_fraction, rng_seed=rng_seed+repetition)
        elif dataset_A == 'bbbc+moa':
            datasets = load_dataset(dataset_A,reduce=5,samples=None, frac=training_data_fraction, rng_seed=rng_seed+repetition)
        elif dataset_A == 'bbbc+comp':
            datasets = load_dataset(dataset_A,reduce=5,samples=None, frac=training_data_fraction, rng_seed=rng_seed+repetition)
        elif dataset_A == '20news_4':
            datasets = load_dataset(dataset_A,reduce=5,samples=None, frac=training_data_fraction, rng_seed=rng_seed+repetition)
        else:
            datasets = load_dataset(dataset_A,reduce=3,samples=None, frac=training_data_fraction, rng_seed=rng_seed+repetition)
                   
        #datasets = load_dataset(dataset_A,reduce=3,samples=None, frac=training_data_fraction, rng_seed=rng_seed)
        #datasets = load_dataset(dataset_A)
        #datasets = load_dataset(dataset_A,reduce=1,samples=100)
    elif source_reuse_mode is 'Join':
        datasets = combine_two_dataset(dataset_A, dataset_B,frac=training_data_fraction, rng_seed=rng_seed+repetition)

    train_set_x, train_set_y = datasets[0]
    #print 'train_set_y', train_set_y
    valid_set_x, valid_set_y = datasets[1]
    test_set_x, test_set_y = datasets[2]



    #datasets2 = load_dataset('csip_sv_ds_ar1',reduce=3,samples=None, frac=training_data_fraction, rng_seed=rng_seed+repetition)
    #datasets2 = load_dataset('csip_ds_ar1',reduce=3,samples=None, frac=training_data_fraction, rng_seed=rng_seed+repetition)
    #train_set_x2, train_set_y2 = datasets2[0]
    
    # compute number of minibatches for training, validation and testing
    #n_train_batches2 = train_set_x2.get_value(borrow=True).shape[0] / batch_size
    n_train_batches = train_set_x.get_value(borrow=True).shape[0] / batch_size
    n_valid_batches = valid_set_x.get_value(borrow=True).shape[0] / batch_size
    n_test_batches = test_set_x.get_value(borrow=True).shape[0] / batch_size
     
    #if approach == 'BL':
    if retrain == 0:    
        # numpy random generator
        #numpy_rng = numpy.random.RandomState(rng_seed);
        numpy_rng = numpy.random.RandomState(rng_seed+repetition)
    
        print '... building the model'
        # construct the stacked denoising autoencoder class
        sda = SdA6(numpy_rng=numpy_rng, n_ins=n_ins,
                  hidden_layers_sizes=hidden_layers_sizes,
                  n_outs=n_outs, #n_outs_b=n_outs_b,
                  tau=tau)
#         from SdA7 import SdA7
#         sda = SdA7(numpy_rng=numpy_rng, n_ins=n_ins,
#                   hidden_layers_sizes=hidden_layers_sizes,
#                   n_outs=n_outs, #n_outs_b=n_outs_b,
#                   tau=tau)
        
      
    
        #########################
        # PRETRAINING THE MODEL #
        #########################
        print '... getting the pretraining functions'
        pretraining_fns = sda.pretraining_functions(train_set_x=train_set_x,
                                                    batch_size=batch_size, tau=tau)
        
        #print "pre-training tau = ",tau
        print '... pre-training the model'
        start_time = time.clock()
        ## Pre-train layer-wise
        corruption_levels = [.2, .3, .3, .3, .3]
        corruption_levels = [.1, .2, .3, .3, .3]
        #corruption_levels = [0, 0, 0]
        pt_trai_costs_vs_stage = []
        for i in xrange(sda.n_layers):
            # go through pretraining epochs
            pt_trai_costs = []
            for epoch in xrange(pretraining_epochs):
                # go through the training set
                c = []
                
                for batch_index in xrange(n_train_batches):
                    c.append(pretraining_fns[i](index=batch_index,
                              corruption=corruption_levels[i],
                              lr=pretrain_lr))
                    
                print 'Pre-training layer %i, epoch %d, cost ' % (i, epoch),
                print numpy.mean(c)
                pt_trai_costs.append(numpy.mean(c))
                #print 'c', c
            pt_trai_costs_vs_stage.append(pt_trai_costs)
            
        end_time = time.clock()
        pt_time = end_time - start_time
        print >> sys.stderr, ('The pretraining code for file ' +
                              os.path.split(__file__)[1] +
                              ' ran for %.2fm' % ((pt_time) / 60.))
        
#         sda_reuse_pt_model = []
#         for para_copy in sda.params_b:
#             sda_reuse_pt_model.append(para_copy.get_value())

        sda_reuse_pt_model = []
        for para_copy in sda.params:
            sda_reuse_pt_model.append(para_copy.get_value())
 
        
        ########################
        # FINETUNING THE MODEL #
        ########################
    
        # get the training, validation and testing function for the model
        print '... getting the finetuning functions'

    
        start_time_ft = time.clock()
        
        if source_reuse_mode is 'Join' and n_outs is not n_outs_source:
            datasets = load_dataset(dataset_A,reduce=3,samples=None, frac=training_data_fraction, rng_seed=rng_seed+repetition)
            train_set_x, train_set_y = datasets[0]
            valid_set_x, valid_set_y = datasets[1]
            test_set_x, test_set_y = datasets[2]
        
            # compute number of minibatches for training, validation and testing
            n_train_batches = train_set_x.get_value(borrow=True).shape[0] / batch_size
            n_valid_batches = valid_set_x.get_value(borrow=True).shape[0] / batch_size
            n_test_batches = test_set_x.get_value(borrow=True).shape[0] / batch_size
            
            
            
            
        train_fn, validate_model, test_model, test_predictions, test_class_probabilities = sda.build_finetune_functions(
                    datasets=datasets, batch_size=batch_size,
                    learning_rate=finetune_lr)
            
#         train_fn, validate_model, test_model, test_predictions = sda.build_finetune_functions(
#                     datasets=datasets, batch_size=batch_size,
#                     learning_rate=finetune_lr)
        
        
        
        print '... finetunning the model'
        
        best_validation_loss, test_score, test_predict, val_epochs,val_epochs_errs, test_epochs, test_epochs_errs, test_class_prob = train_test_mlp(learning_rate=0.01, training_epochs=training_epochs,#1000,
                         dataset=dataset_A, batch_size=batch_size, 
                         n_train_batches=n_train_batches,n_valid_batches=n_valid_batches,n_test_batches=n_test_batches,
                         train_fn=train_fn,validate_model=validate_model,test_model=test_model, test_predictions=test_predictions,
                         test_class_probabilities=test_class_probabilities)
        
        
        #test_class_prob = numpy.array(test_class_prob)
        #print numpy.shape(test_class_prob)
        
        #print test_class_prob[:,0].argmax(axis=0)
#         y_test_class_prob = test_class_prob[:,0]
#         y_test_class = test_class_prob[:,1]
#         import cPickle as pickle
#         pickle.dump(y_test_class_prob, open('bbbc_y_test_class_prob'+str(repetition)+'.pkl', 'wb'))
#         pickle.dump(y_test_class, open('bbbc_y_test_class.pkl', 'wb'))
#         #print test_class_prob
#         print numpy.shape(y_test_class_prob)
#         print y_test_class_prob[0:1, 0:10]
#         print y_test_class[0:1, 0:10]
        
# #         best_validation_loss, test_score, test_predict, val_epochs,val_epochs_errs, test_epochs, test_epochs_errs = train_test_mlp(learning_rate=0.01, training_epochs=training_epochs,#1000,
# #                          dataset=dataset_A, batch_size=batch_size, 
# #                          n_train_batches=n_train_batches,n_valid_batches=n_valid_batches,n_test_batches=n_test_batches,
# #                          train_fn=train_fn,validate_model=validate_model,test_model=test_model, test_predictions=test_predictions)
        
        test_predict = numpy.array(test_predict)
        y_test_pred = test_predict[:,0]
        y_test = test_predict[:,1]
        
        end_time_ft = time.clock()
        ft_time = end_time_ft - start_time_ft
        
        sda_reuse_ft2_model = []
        for para_copy in sda.params:
            #print 'para_copy22.get_value()',para_copy.get_value()
            sda_reuse_ft2_model.append(para_copy.get_value())
        
#         for ids in range(len(sda.params)):
#             a = sda.params[ids].get_value()
#             print 'a',a
#             sda_reuse_ft2_model.append(a)

        sda_reuse_ft_model = sda
        print 'done'
    
    
    ########################
        # RE- FINETUNING THE MODEL #
    ########################
    elif retrain == 1:
            
            from scipy.stats import bernoulli
            # numpy random generator
            numpy_rng = numpy.random.RandomState(rng_seed+repetition)
            
            print '... building the model'
            # construct the stacked denoising autoencoder class
            sda = SdA6(numpy_rng=numpy_rng, n_ins=n_ins,
                      hidden_layers_sizes=hidden_layers_sizes,
                      n_outs=n_outs, #n_outs_b=n_outs_b,
                      tau=tau)
            
            if source_reuse_mode == 'R':
                print 'random initialization'
                
            elif source_reuse_mode == 'R+D':
                print 'random initialization with dropout'
                for ids in range(len(sda.params)):
                    a = sda.params[ids].get_value()
                    b = dropout_weights(a, dropout_rate)
                    sda.params[ids].set_value(b)
                    
            elif source_reuse_mode == 'PT':
                print 'restoring source problem pre-training weights'
                if n_outs == n_outs_source:
                    for ids in range(len(sda.params)):
                        sda.params[ids].set_value(sda_reuse_pt_model[ids]) # set the value
                else:
                    for ids in range(len(sda.params)-2):
                        sda.params[ids].set_value(sda_reuse_pt_model[ids]) # set the value
            
            elif source_reuse_mode == 'PT+D':
                print 'restoring source problem pre-training weights with dropout'
                if n_outs == n_outs_source:
                    for ids in range(len(sda.params)):
                        a = sda_reuse_pt_model[ids]
                        b = dropout_weights(a, dropout_rate)
                        sda.params[ids].set_value(b)                       
                else:
                    for ids in range(len(sda.params)-2):
                        a = sda_reuse_pt_model[ids]
                        b = dropout_weights(a, dropout_rate)
                        sda.params[ids].set_value(b)      
            
            elif source_reuse_mode == 'PT+FT':
                print 'restoring source problem fine-tunned weights'
                if n_outs == n_outs_source:
                    for ids in range(len(sda.params)):
                        sda.params[ids].set_value(sda_reuse_ft_model.params[ids].get_value())
                else:
                    for ids in range(len(sda.params)-2):
                        #sda.params[ids].set_value(sda_reuse_ft_model.params[ids].get_value())
                        
                        # FOR BBBC data from float64 to float32
                        a = sda_reuse_ft_model.params[ids].get_value() 
                        b = a.astype(dtype='float32') 
                        sda.params[ids].set_value(b)
            
            elif source_reuse_mode == 'PT+FT+D':
                print 'restoring source problem fine-tunned weights with dropout'
                if n_outs == n_outs_source:
                    for ids in range(len(sda.params)):
                        a = sda_reuse_ft_model.params[ids].get_value()
                        b = dropout_weights(a, dropout_rate)
                        sda.params[ids].set_value(b)
                else:
                    for ids in range(len(sda.params)-2):
                        a = sda_reuse_ft_model.params[ids].get_value()
                        b = dropout_weights(a, dropout_rate)

                        sda.params[ids].set_value(b)
            
                        
            
            start_time_rft = time.clock()
            train_fn, validate_model, test_model, test_predictions, test_class_probabilities = sda.build_finetune_functions(
                    datasets=datasets, batch_size=batch_size,
                    learning_rate=finetune_lr)
            
            print '... finetunning the model'
        
            best_validation_loss, test_score, test_predict, val_epochs,val_epochs_errs, test_epochs, test_epochs_errs, test_class_prob = train_test_mlp(learning_rate=0.01, training_epochs=training_epochs,#1000,
                         dataset=dataset_A, batch_size=batch_size, 
                         n_train_batches=n_train_batches,n_valid_batches=n_valid_batches,n_test_batches=n_test_batches,
                         train_fn=train_fn,validate_model=validate_model,test_model=test_model, test_predictions=test_predictions,
                         test_class_probabilities=test_class_probabilities)
            
            
            
#             train_fn, validate_model, test_model, test_predictions = sda.build_finetune_functions_reuse(
#                         datasets=datasets, batch_size=batch_size,
#                         learning_rate=finetune_lr, retrain_ft_layers= retrain_ft_layers)
#             
#             best_validation_loss, test_score, test_predict, val_epochs,val_epochs_errs, test_epochs, test_epochs_errs = train_test_mlp(learning_rate=0.01, training_epochs=training_epochs,
#                      dataset=dataset_A, batch_size=batch_size,
#                      n_train_batches=n_train_batches,n_valid_batches=n_valid_batches,n_test_batches=n_test_batches,
#                      train_fn=train_fn,validate_model=validate_model,test_model=test_model, test_predictions=test_predictions)
            
            end_time_rft = time.clock()
            pt_time = 0
            ft_time = end_time_rft - start_time_rft
            
            test_predict = numpy.array(test_predict)
            y_test_pred = test_predict[:,0]
            y_test = test_predict[:,1]
        
            sda_reuse_ft_model = sda
            sda_reuse_ft2_model = []
            for para_copy in sda.params:
                sda_reuse_ft2_model.append(para_copy.get_value())
            sda_reuse_pt_model= None
            pt_trai_costs_vs_stage = None

    return (sda_reuse_pt_model,sda_reuse_ft2_model, sda_reuse_ft_model,
            best_validation_loss*100, test_score*100, 
            pt_time, ft_time, y_test_pred, y_test, 
            val_epochs,val_epochs_errs, test_epochs, test_epochs_errs,
            pt_trai_costs_vs_stage)