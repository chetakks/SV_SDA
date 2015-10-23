from source_t_source3 import reuse_SdA6
from numpy import mean,std
import datetime
import cPickle
import pickle
import os
import sys
import time
import csv 
import load_dataset2

def run_n_times(params,nr_reps,target_dataset,source_dataset,training_data_fraction,
                source_outputs_dir,target_outputs_dir,source_reuse_mode, retrain_ft_layers,
                approach = None, transfer = None):
    
    
    if not os.path.exists(target_outputs_dir):
            os.makedirs(target_outputs_dir)
    print 'target_outputs_dir',target_outputs_dir
    print 'Fetching target dataset information ...'
    if target_dataset == 'mnist_64x80':
        #[n_ins, n_outs]     = load_dataset2.dataset_details(target_dataset, reduce=5)
        [n_ins, n_outs]     = [5120, 10]
    elif target_dataset == 'bbbc':
        #[n_ins, n_outs]     = load_dataset2.dataset_details(target_dataset)
        [n_ins, n_outs]     = [5120, 3]
    elif target_dataset == 'bbbc+feat':
        #[n_ins, n_outs]     = load_dataset2.dataset_details(target_dataset)
        [n_ins, n_outs]     = [453, 13]
    elif target_dataset == 'bbbc+feat2':
        #[n_ins, n_outs]     = load_dataset2.dataset_details(target_dataset)
        [n_ins, n_outs]     = [453, 13]
    elif target_dataset == 'bbbc+feat3':
        [n_ins, n_outs]     = [453, 12]
    elif target_dataset == 'bbbc+moa':
        [n_ins, n_outs]     = [453, 12]
    elif target_dataset == 'bbbc+comp':
        [n_ins, n_outs]     = [453, 38]
    elif target_dataset == 'mnist':
        [n_ins, n_outs]     = [784, 10]
    elif target_dataset == 'chars74k_uppercase28x28':
        [n_ins, n_outs]     = [784, 26]
    elif target_dataset == 'chars74k_lowercase28x28':
        [n_ins, n_outs]     = [784, 26]
    elif target_dataset == '20news_4':
        [n_ins, n_outs]     = [33810, 4]
    else:
        [n_ins, n_outs]     = load_dataset2.dataset_details(target_dataset)
    #n_ins, n_outs = 784, 26
    params['dataset_A'] = target_dataset
    params['n_ins']     = n_ins
    params['n_outs']    = n_outs
    params['dataset_B'] = None
    n_ins_source = None
    params['n_outs_source']  = None
    dropout = None
    dropout_rate = None
    print 'target_dataset details', params['dataset_A']
    print '                 n_ins', params['n_ins']
    print '                n_outs', params['n_outs']
    print '               dropout', dropout
    print '          dropout_rate', dropout_rate
             
    if source_outputs_dir is not None and source_reuse_mode is not 'Join':
        # load source networks
        #print 'load source model from ',source_outputs_dir
        source_outputs_list = load_dataset2.load_outputs(source_outputs_dir)
        
        
        if retrain_ft_layers is not None:
            retrain_ft_layers_tmp = []
            print 'retrain_ft_layers = ',retrain_ft_layers
            for x in retrain_ft_layers:
                retrain_ft_layers_tmp.extend([int(x), int(x)])   
            retrain_ft_layers = retrain_ft_layers_tmp   
            #print 'retrain_ft_layers = ',retrain_ft_layers
        
        print 'Fetching source dataset information ...'
        if source_dataset == 'mnist_64x80':
            #[n_ins_source, n_outs_source]     = load_dataset2.dataset_details(source_dataset, reduce=5)
            [n_ins_source, n_outs_source]     = [5120, 10]
        elif source_dataset == 'bbbc':
            #[n_ins_source, n_outs_source]     = load_dataset2.dataset_details(source_dataset)
            [n_ins_source, n_outs_source]     = [5120, 3]
        elif source_dataset == 'bbbc+feat':
            #[n_ins_source, n_outs_source]     = load_dataset2.dataset_details(source_dataset)
            [n_ins_source, n_outs_source]     = [453, 13]
        elif source_dataset == 'bbbc+feat2':
            #[n_ins_source, n_outs_source]     = load_dataset2.dataset_details(source_dataset)
            [n_ins_source, n_outs_source]     = [453, 13]
        elif source_dataset == 'bbbc+feat3':
            #[n_ins_source, n_outs_source]     = load_dataset2.dataset_details(source_dataset)
            [n_ins_source, n_outs_source]     = [453, 12]
        elif source_dataset == 'bbbc+moa':
            [n_ins_source, n_outs_source]     = [453, 12]
        elif source_dataset == 'bbbc+comp':
            [n_ins_source, n_outs_source]     = [453, 38]
        elif source_dataset == 'mnist':
            [n_ins_source, n_outs_source]     = [784, 10]
        elif source_dataset == 'chars74k_uppercase28x28':
            [n_ins_source, n_outs_source]     = [784, 26]
        elif source_dataset == 'chars74k_lowercase28x28':
            [n_ins_source, n_outs_source]     = [784, 26]
        elif source_dataset == '20news_4':
            [n_ins_source, n_outs_source]     = [33810, 4]
        else:
            [n_ins_source, n_outs_source]     = load_dataset2.dataset_details(source_dataset)
      
        #n_ins_source, n_outs_source = 784, 10
        params['dataset_B'] = source_dataset
        params['n_outs_source']  = n_outs_source
        dropout = None
        dropout_rate = 0.5
        print 'source_dataset details', params['dataset_B']
        print '          n_ins_source', n_ins_source
        print '         n_outs_source', params['n_outs_source']
        
    elif source_outputs_dir is not None and source_reuse_mode is 'Join':
        print 'Fetching source dataset information ...'
        [n_ins_source, n_outs_source] = load_dataset2.dataset_details(source_dataset)
        #n_ins_source, n_outs_source = 784, 10
        params['dataset_B'] = source_dataset
        params['n_outs_source']  = n_outs_source
        dropout = None
        dropout_rate = 0.5
        print 'source_dataset details', params['dataset_B']
        print '          n_ins_source', n_ins_source
        print '         n_outs_source', params['n_outs_source']
        

        
    timestamp = datetime.datetime.now().strftime('%Y%m%d%H%M%S')
    #print timestamp
    
    
    
       

    #file_name = 'train_' +  params['dataset_A']

    tau = None     
    sda_reuse_pt_models = []
    sda_reuse_ft2_models = []
    sda_reuse_ft_models = []
    best_val_errors = []
    test_score_errors = []
    pt_times = []
    ft_times = []
    y_test_preds = []
    y_tests = []
    val_epochs_rep = []
    val_epochs_errs_rep = []
    test_epochs_rep = []
    test_epochs_errs_rep = []
    
    if target_dataset == 'bbbc+moa' or target_dataset == 'bbbc+comp':
        avg_accs = []
        avg_cms = []
        ind_accs = []
        ind_cms = []
    
    start_tune_time = time.clock()
    
    
    for repetition in range(nr_reps):
        print '------------repetition: ', repetition+1
        output_file_path = target_outputs_dir+'outputs_'+timestamp+'_%03d.pkl.gz' % (repetition+1)
        print output_file_path

        if source_outputs_dir is not None and source_reuse_mode is not 'Join':
            print 'Restoration of weights of source dataset' 
            #task = 'Reuse '+ source_dataset + ' layers for classifying ' + target_dataset + ' mode ' +source_reuse_mode
            task = 'TL: '+ target_dataset + ' reusing ' + str(source_dataset) +  ' mode ' + str(source_reuse_mode) +' layers '+ str(retrain_ft_layers)
            print task
            
            source_outputs = source_outputs_list[repetition]
            if   source_reuse_mode == 'PT' or 'PT+D':
                sda_reuse_pt_model = source_outputs['sda_reuse_pt_model']
                sda_reuse_ft_model = source_outputs['sda_reuse_ft_model']
            elif source_reuse_mode == 'PT+FT' or 'PT+FT+D':
                sda_reuse_pt_model = None
                sda_reuse_ft_model = source_outputs['sda_reuse_ft_model']

        else:
            task = 'Experiment with ' + target_dataset + ' and ' + str(source_dataset) +  ' mode ' + str(source_reuse_mode) 
            print task
            sda_reuse_ft_model = None
            sda_reuse_pt_model = None
            
        sda_reuse_pt_model,sda_reuse_ft2_model,sda_reuse_ft_model,best_val_error, test_score, pt_time, ft_time, y_test_pred, y_test, val_epochs_rep,val_epochs_errs_rep, test_epochs_rep, test_epochs_errs_rep, pt_trai_costs_vs_stage = reuse_SdA6(    
        #sda_reuse_pt_model,sda_reuse_ft_model, best_val_error, test_score, pt_time, ft_time = reuse_SdA6(
             params['finetune_lr'],
             params['pretraining_epochs'],
             params['pretrain_lr'],
             params['training_epochs'],
             params['n_ins'],
             params['hidden_layers_sizes'],
             params['dataset_A'],
             params['n_outs'],
             params['retrain'],
             source_reuse_mode,
             #params['reset_pt'],
             params['dataset_B'],
             params['n_outs_source'],
             #params['n_outs_b'],
             params['batch_size'],
             params['output_fold'],
             params['rng_seed'],
             retrain_ft_layers = retrain_ft_layers,
             sda_reuse_pt_model=sda_reuse_pt_model,
             sda_reuse_ft_model=sda_reuse_ft_model,
             repetition=repetition,
             tau=tau,
             training_data_fraction=training_data_fraction,
             #dropout = dropout,
             dropout_rate = dropout_rate)  
        sda_reuse_pt_models.append(sda_reuse_pt_model)
        sda_reuse_ft2_models.append(sda_reuse_ft2_model)
        sda_reuse_ft_models.append(sda_reuse_ft_model)
        best_val_errors.append(best_val_error)
        test_score_errors.append(test_score)
        pt_times.append(pt_time)
        ft_times.append(ft_time)
        y_test_preds.append(y_test_pred)
        y_tests.append(y_test)
        
        
        
        
        outputs = {}
    
        # parameters
        #outputs['cpu_model_name']   = cpuinfo.cpu.info[0]['model name']
        outputs['hidden_sizes']     = params['hidden_layers_sizes']
        outputs['pt_learning_rate'] = params['pretrain_lr']
        #outputs['pt_noise_prob']    = pt_noise_prob
        #outputs['pt_cost_function'] = pt_cost_function
        #outputs['pt_tau']           = pt_tau
        outputs['ft_learning_rate'] = params['finetune_lr']
        outputs['ft_look_ahead']    = params['training_epochs']
        #outputs['ft_cost_function'] = ft_cost_function
        #outputs['ft_tau']           = ft_tau
        
        # performance measures
    
        # pre-training mean reconstruction costs over validation and test sets 
        #outputs['pt_vali_costs']   = pt_vali_costs
        #outputs['pt_test_costs']   = pt_test_costs
        # fine-tuning validation error (lowest)
        outputs['ft_vali_err']     = best_val_error
        # fine-tuning test error
        outputs['ft_test_err']     = test_score
        # fine-tuning training error
        #outputs['ft_trai_err']     = ft_trai_err
        # fine-tuning balanced test error
        #outputs['ft_bal_test_err'] = ft_bal_test_err
            
        # the pre-trained model
        outputs['sda_reuse_pt_model'] = sda_reuse_pt_model
        # the fine-tuned model
        outputs['sda_reuse_ft_model'] = sda_reuse_ft_model
        
        # pre-training time for each hidden layer
        outputs['pt_trai_times'] = pt_time
        # fine-tuning time
        outputs['ft_trai_time']  = ft_time
        
        outputs['y_test_pred'] = y_test_pred
        outputs['y_test'] = y_test
        
        outputs['val_epochs_rep']     = val_epochs_rep
        outputs['val_epochs_errs_rep']     = val_epochs_errs_rep
        outputs['test_epochs_rep'] = test_epochs_rep
        outputs['test_epochs_errs_rep'] = test_epochs_errs_rep
        outputs['pt_trai_costs_vs_stage']   = pt_trai_costs_vs_stage
        
        if target_dataset == 'bbbc+moa' or target_dataset == 'bbbc+comp':
            from prepare_bbbc_results import bbbc_collective_results
            avg_acc, avg_cm, ind_acc, ind_cm = bbbc_collective_results(target_dataset, y_test_pred, y_test)
            outputs['avg_acc'] = avg_acc
            outputs['avg_cm'] = avg_cm
            outputs['ind_acc'] = ind_acc
            outputs['ind_cm'] = ind_cm
            avg_accs.append(avg_acc)
            avg_cms.append(avg_cm)
            ind_accs.append(ind_acc)
            ind_cms.append(ind_cm)
        
            
                    
        # ft validation error vs. epoch
        #outputs['ft_vali_err_vs_stage'] = ft_vali_err_vs_stage
        # this can be used e.g. to obtain the error for each class
        #outputs['ft_confusion_matrix']  = ft_confusion_matrix
#         save_training_info = 1 
#         if save_training_info:
#             # save extra information on evolution of training
#              
#             outputs['pt_trai_costs_vs_stage']   = pt_trai_costs_vs_stage
#             outputs['pt_vali_costs_vs_stage']   = pt_vali_costs_vs_stage
#             outputs['pt_test_costs_vs_stage']   = pt_test_costs_vs_stage
#                     
#             outputs['ft_trai_err_vs_stage']     = ft_trai_err_vs_stage
#             outputs['ft_test_err_vs_stage']     = ft_test_err_vs_stage
#             outputs['ft_n_incr_error_vs_stage'] = ft_n_incr_error_vs_stage
#             outputs['ft_effective_lr_vs_stage'] = ft_effective_lr_vs_stage
        print 'output_file_path', output_file_path
        load_dataset2.save(outputs,output_file_path)
        
        save_training_info = 0
        if save_training_info:
            # save information on weights and biases
#             if not os.path.exists(target_outputs_dir):
#                 os.makedirs(target_outputs_dir)
            import scipy.io as sio
            print 'saving info PT weights'
            sio.savemat(target_dataset+'_PT_WB.mat', {'WB':sda_reuse_pt_model})
            print 'saving info FT weights'
            sio.savemat(target_dataset+'_FT_WB.mat', {'WB':sda_reuse_ft2_model})   

                  
        
        
        print 'Result of repetition # ', repetition + 1
        print 'training data fraction =' +str(training_data_fraction)
        if target_dataset == 'bbbc+moa' or target_dataset == 'bbbc+comp':  
            #print 'Classification results of ', target_dataset
            print 'Collective the prediction of individual cells'
            print 'Coll Accuracy =' + str(avg_accs)
            print 'Coll mean Accuracy = %.2f(%.2f)' % (mean(avg_accs)*100, std(avg_accs)*100)  
            print 'Prediction of individual cells'
            print 'Ind Accuracy =' + str(ind_accs)
            print 'Ind mean Accuracy = %.2f(%.2f)' % (mean(ind_accs)*100, std(ind_accs)*100) 
        else: 
            print 'best_validation error =' + str(best_val_errors)
            print 'mean validation error =' + str(mean(best_val_errors))
            print 'std validation error =' + str(std(best_val_errors))
            print 'Test error =' + str(test_score_errors)
            print 'mean test error =' + str(mean(test_score_errors))
            print 'std test error =' + str(std(test_score_errors))   
        print 'Time take for train pt layers in sec = ', pt_times
        print 'Time to train pt layers = mean %.2f(%.2f)s' % (mean(pt_times), std(pt_times))
        print 'Time take for train ft layers in sec = ', ft_times
        print 'Time to train ft layers = mean %.2f(%.2f)s' % (mean(ft_times), std(ft_times))
        #from sklearn.metrics import confusion_matrix
        #print confusion_matrix(y_test.flatten(), y_test_pred.flatten())
   
       
        
    
        if repetition == (nr_reps-1): 
            fm = open(str(target_outputs_dir)+'Results_grid.txt','a')
            old_stdout = sys.stdout   
            sys.stdout = fm
            print '===================================================================='
#             print 'Approach' + approach
#             if transfer == None:
#                 print 'approach', approach  
#             elif transfer % 2 == 0: # even for STS
#                 print 'even number of transfers, approach = STS :', 'STS_'+str(transfer / 2)
#             elif transfer % 2 == 1: # odd for TL
#                 print 'odd number of transfers,  approach = TL :', 'TL_'+str((transfer / 2)+1)
#             
#             print 'training data fraction ='+ str(training_data_fraction)
            print 'target_dataset details: '+ str(target_dataset)
            print '            features  = '+ str(params['n_ins']) 
            print '             targets  = '+ str(params['n_outs'])
#             print 'source_dataset details: '+ str(source_dataset)
#             print '             features = '+ str(n_ins_source)
#             print '              targets = '+ str(params['n_outs_source'])
            print 'Architecture details:   '
            print 'hidden_layers_sizes= '+ str(params['hidden_layers_sizes'])
            print 'Max Nr. PT epochs  = '+ str(params['pretraining_epochs'])
            print 'Max Nr. FT epochs  = '+ str(params['training_epochs'])
            print 'PT learning_rate   = '+ str(params['pretrain_lr'])
            print 'FT learning_rate   = '+ str(params['finetune_lr'])
            print 'batch_size         = '+ str(params['batch_size'])
#             print 'dropout            = '+ str(dropout)
#             print 'dropout_rate       = '+ str(dropout_rate)
#             print 'source_reuse_mode  = '+ str(source_reuse_mode)
            print
            print '====  results for repetition # ' + str(repetition+1) + ' out of '+ str(nr_reps)
            if source_reuse_mode == 'R' or source_reuse_mode == 'R+D' or source_reuse_mode == 'PT+FT' or source_reuse_mode == 'PT+FT+D':
                print 'layers retrained by dataset_A : ', retrain_ft_layers
                print 'Time to train ft layers = mean %.2f(%.2f)s' % (mean(ft_times), std(ft_times))
            elif  source_reuse_mode == 'PT' or source_reuse_mode == 'PT+D':
                print 'Time to train pt layers = mean %.2f(%.2f)s' % (mean(pt_times), std(pt_times))
                print 'Time to train ft layers = mean %.2f(%.2f)s' % (mean(ft_times), std(ft_times))
            else:
                print 'Time to train pt layers = mean %.2f(%.2f)s' % (mean(pt_times), std(pt_times))
                print 'Time to train ft layers = mean %.2f(%.2f)s' % (mean(ft_times), std(ft_times))
            if target_dataset == 'bbbc+moa' or target_dataset == 'bbbc+comp':  
                #print 'Classification results of ', target_dataset
                print 'Collective the prediction of individual cells'
                print '     Accuracy =' + str(avg_acc)
                print '     mean Accuracy = %.2f(%.2f)' % (mean(avg_accs)*100, std(avg_accs)*100)  
                print 'Prediction of individual cells'
                print '     Accuracy =' + str(ind_acc)
                print '     mean Accuracy = %.2f(%.2f)' % (mean(ind_accs)*100, std(ind_accs)*100) 
                print
            else: 
                print 'Test error =' + str(test_score_errors)
                print 'mean test error = %.2f(%.2f)' % (mean(test_score_errors), std(test_score_errors)) 
                print
            sys.stdout=old_stdout 
            fm.close()
            
        if repetition == (nr_reps-1): 
            f2 = open('Grid_results_sv_ds/Master_grid_results.txt','a')
            old_stdout = sys.stdout   
            sys.stdout = f2
            print '===================================================================='
            print 'target_dataset details: '+ str(target_dataset)
            print '            features  = '+ str(params['n_ins']) 
            print '             targets  = '+ str(params['n_outs'])
            print 'Architecture details:   '
            print 'hidden_layers_sizes= '+ str(params['hidden_layers_sizes'])
            print 'Max Nr. PT epochs  = '+ str(params['pretraining_epochs'])
            print '====  results for repetition # ' + str(repetition+1) + ' out of '+ str(nr_reps)
            print 'Time to train pt layers = mean %.2f(%.2f)s' % (mean(pt_times), std(pt_times))
            print 'Time to train ft layers = mean %.2f(%.2f)s' % (mean(ft_times), std(ft_times))
            print 'Test error =' + str(test_score_errors)
            print 'mean test error = %.2f(%.2f)' % (mean(test_score_errors), std(test_score_errors)) 
            print
            sys.stdout=old_stdout 
            f2.close()
            
            if approach == 'MSTS':
                fm2 = open(str(target_outputs_dir)+'Master_STS.txt','a')
                #fm2 = open('Master_STS_[1111]_gpu0.txt','a') 
                #fm2 = open('Master_STS_BBBC[1111]_gpu0.txt','a')
                #fm2 = open('Master_STS_BBBC[1111]_gpu0_20rep.txt','a')
                #fm2 = open('Master_STS_BBBC[1111]com_gpu0_20rep.txt','a')
                #fm2 = open('Master_STS_BBBC[0011]moa+com_gpu0_20rep.txt','a')
                #fm2 = open('Master_STS_BBBC[0011]_gpu0.txt','a')
                #fm2 = open('Master_STS_BBBC[0011]com_gpu0.txt','a')
                old_stdout = sys.stdout   
                sys.stdout = fm2 
                print '===================================================================='
                print 'Approach ' + approach
                if transfer == None:
                    print 'approach ', approach  
                elif transfer % 2 == 0: # even for STS
                    print 'even number of transfers, approach = STS :', 'STS_'+str(transfer / 2)
                    print 'target_dataset details: '+ str(target_dataset)
                    print 'source_dataset details: '+ str(source_dataset)
                elif transfer % 2 == 1: # odd for TL
                    print 'odd number of transfers,  approach = TL :', 'TL_'+str((transfer / 2)+1)
                    print 'target_dataset details: '+ str(target_dataset)
                    print 'source_dataset details: '+ str(source_dataset)
                

                print 'source_reuse_mode  = '+ str(source_reuse_mode)
                print
                print '====  results for repetition # ' + str(repetition+1) + ' out of '+ str(nr_reps)
                if source_reuse_mode == 'R' or 'R+D' or 'PT+FT' or 'PT+FT+D':
                    print 'layers retrained by dataset_A : ', retrain_ft_layers
                    print 'Time to train ft layers = mean %.2f(%.2f)s' % (mean(ft_times), std(ft_times))
                elif  source_reuse_mode == 'PT' or 'PT+D':
                    print 'Time to train pt layers = mean %.2f(%.2f)s' % (mean(pt_times), std(pt_times))
                    print 'Time to train ft layers = mean %.2f(%.2f)s' % (mean(ft_times), std(ft_times))
                else:
                    print 'Time to train pt layers = mean %.2f(%.2f)s' % (mean(pt_times), std(pt_times))
                    print 'Time to train ft layers = mean %.2f(%.2f)s' % (mean(ft_times), std(ft_times))
                    
                if target_dataset == 'bbbc+moa' or target_dataset == 'bbbc+comp':  
                    #print 'Classification results of ', target_dataset
                    print 'Collective the prediction of individual cells'
                    print '     Accuracy =' + str(avg_acc)
                    print '     mean Accuracy = %.2f(%.2f)' % (mean(avg_accs)*100, std(avg_accs)*100)  
                    print 'Prediction of individual cells'
                    print '     Accuracy =' + str(ind_acc)
                    print '     mean Accuracy = %.2f(%.2f)' % (mean(ind_accs)*100, std(ind_accs)*100) 
                    print
                else: 
                    print 'Test error =' + str(test_score_errors)
                    print 'mean test error = %.2f(%.2f)' % (mean(test_score_errors), std(test_score_errors)) 
                    print
                sys.stdout=old_stdout 
                fm2.close()
        
    end_tune_time = time.clock()
    
    print 'The code ran for %.2fm' % ((end_tune_time - start_tune_time) / 60.)
    
    print >> sys.stderr, ('Testing the Reusability SDA code for file ' +
                       os.path.split(__file__)[1] +
                       ' ran for %.2fm' % ((end_tune_time - start_tune_time) / 60.))