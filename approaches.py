import os, sys
from repetitions import run_n_times


def bl_or_tl(experiment,training_data_fraction,results_dir,params,nr_reps):
    [approach, target_dataset, source_dataset, source_reuse_mode, retrain_ft_layers] = experiment
       
    if approach == 'BL':
        params['retrain'] = 0
        params['reset_pt'] = 0
        #print experiment
        
        if source_reuse_mode == None:
        
            source_outputs_dir = None
            target_outputs_dir = '{0}{1}_{2:4.2f}/'\
            .format(results_dir,
                    target_dataset,
                    training_data_fraction)
            
        elif source_reuse_mode == 'Join':
            
            source_outputs_dir = source_dataset
            target_outputs_dir = '{0}{1}_{2}_{3}_{4:4.2f}/'\
            .format(results_dir,
                    target_dataset,
                    source_reuse_mode,
                    source_dataset,
                    training_data_fraction)
        
        run_n_times(params, nr_reps,target_dataset,source_dataset,
                    training_data_fraction,source_outputs_dir,target_outputs_dir,source_reuse_mode, retrain_ft_layers,
                    approach)
        
    elif approach == 'Tune':
        print 'BL with different architectures approach for parameter search'
        params['retrain'] = 0
        params['reset_pt'] = 0
        nn = params['hidden_layers_sizes']
        pt = params['pretraining_epochs']
        print "params['hidden_layers_sizes']", nn
        print "params['pretraining_epochs']",  pt
        #print experiment
        if source_reuse_mode == None:
            source_outputs_dir = None
            target_outputs_dir = '{0}{1}{2}{3}_{4:5.2f}/'\
            .format(results_dir,
                    target_dataset,
                    nn,
                    pt,
                    training_data_fraction)   
        
        print target_outputs_dir
        #settings['source_outputs_dir'] = source_outputs_dir
        #settings['target_outputs_dir'] = target_outputs_dir
        #run_n_times(settings,params)
        
        run_n_times(params, nr_reps,target_dataset,source_dataset,
                    training_data_fraction,
                    source_outputs_dir,target_outputs_dir,source_reuse_mode, 
                    retrain_ft_layers,
                    approach)
        
            
    elif approach == 'TL':
            params['retrain'] = 1
            #retrain = 1
            if source_reuse_mode == 'PT+FT':
                params['reset_pt']= 0
            elif source_reuse_mode == 'PT+FT+D':
                params['reset_pt']= 0
                #dropout = 1
                #dropout_rate = 0.5
            elif source_reuse_mode == 'PT':
                params['reset_pt']= 1
            elif source_reuse_mode == 'PT+D':
                params['reset_pt']= 1
                #dropout = 1
                #dropout_rate = 0.5
            elif source_reuse_mode == 'R':
                params['reset_random']= 1
                #dropout = 1
                #dropout_rate = 0.5
                
            #print experiment
            

            source_outputs_dir = '{0}{1}'\
                    .format(results_dir,
                            source_dataset)
            source_outputs_dir = source_outputs_dir+'_1.00/'
            target_outputs_dir = '{0}{1}_reusing_{2}_{3}_{4}_{5:4.2f}/'\
                 .format(results_dir,
                         target_dataset,
                         source_dataset,
                         source_reuse_mode,
                         str(retrain_ft_layers),
                         training_data_fraction)
                 
            run_n_times(params, nr_reps,target_dataset,source_dataset,
                    training_data_fraction,source_outputs_dir,target_outputs_dir,source_reuse_mode, retrain_ft_layers,
                    approach)
  
            
    elif approach == 'STS':
            params['retrain']  = 1
            if source_reuse_mode == 'PT+FT':
                params['reset_pt']= 0
            elif source_reuse_mode == 'PT':
                params['reset_pt']= 1
            #print experiment
             
            [TL_approach, TL_target_dataset, TL_source_dataset, TL_source_reuse_mode, TL_retrain_ft_layers] = source_dataset
            STS_source_dataset = source_dataset  #back up 
#             source_outputs_dir = '{0}{1}_reusing_{2}_{3}_{4}_{5:4.2f}/'\
#                  .format(results_dir,
#                          TL_target_dataset,
#                          TL_source_dataset,
#                          TL_source_reuse_mode,
#                          str(TL_retrain_ft_layers),
#                          training_data_fraction)

            source_outputs_dir = '{0}{1}_reusing_{2}_{3}_{4}'\
                 .format(results_dir,
                         TL_target_dataset,
                         TL_source_dataset,
                         TL_source_reuse_mode,
                         str(TL_retrain_ft_layers))
            source_outputs_dir = source_outputs_dir+'_1.00/'
                 
                 
            target_outputs_dir = '{0}{1}_reusing_{2}_{3}_{4}_{5:4.2f}/'\
                 .format(results_dir,
                         target_dataset,
                         source_dataset,
                         source_reuse_mode,
                         str(retrain_ft_layers),
                         training_data_fraction)
            
            source_dataset=TL_target_dataset
            run_n_times(params, nr_reps,target_dataset,source_dataset,
                    training_data_fraction,source_outputs_dir,target_outputs_dir,source_reuse_mode, retrain_ft_layers,
                    approach)
            
    
    elif approach == 'MSTS':
            num_of_STS = 10
            #num_of_STS = 3
            total_transfers = num_of_STS *2 
            
            params['retrain'] = 1
            #retrain = 1
            if source_reuse_mode == 'PT+FT':
                params['reset_pt']= 0
            elif source_reuse_mode == 'PT+FT+D':
                params['reset_pt']= 0
                #dropout = 1
                #dropout_rate = 0.5
            elif source_reuse_mode == 'PT':
                params['reset_pt']= 1
            elif source_reuse_mode == 'PT+D':
                params['reset_pt']= 1
                #dropout = 1
                #dropout_rate = 0.5
            elif source_reuse_mode == 'R':
                params['reset_random']= 1
                #dropout = 1
                #dropout_rate = 0.5
            
            for transfer in range(1,total_transfers+1):
                #print transfer
                if transfer == 1: # odd for TL
                    print 'odd number of transfers,  approach = TL :', 'TL_'+str((transfer / 2)+1)
                    source_outputs_dir = '{0}{1}'\
                            .format(results_dir,
                                    source_dataset)
                    source_outputs_dir = source_outputs_dir+'_1.00/'
                    target_outputs_dir = '{0}{1}_reusing_{2}_TL{3}/'\
                         .format(results_dir,
                                 target_dataset,
                                 source_dataset,
                                 str(transfer))
            
                    print 'source_outputs_dir', source_outputs_dir
                    print 'target_outputs_dir', target_outputs_dir
                          
                    run_n_times(params, nr_reps,target_dataset,source_dataset,
                            training_data_fraction,source_outputs_dir,target_outputs_dir,source_reuse_mode, retrain_ft_layers,
                            approach, transfer)
            
                    
                else:
                    if transfer % 2 == 0: # even for STS
                        print 'even number of transfers, approach = STS :', 'STS_'+str(transfer / 2)
                    elif transfer % 2 == 1: # odd for TL
                        print 'odd number of transfers,  approach = TL :', 'TL_'+str((transfer / 2)+1)
                        
                    params['retrain'] = 1
                    #retrain = 1
                    if source_reuse_mode == 'PT+FT':
                        params['reset_pt']= 0
                    elif source_reuse_mode == 'PT+FT+D':
                        params['reset_pt']= 0
                        #dropout = 1
                        #dropout_rate = 0.5
                    elif source_reuse_mode == 'PT':
                        params['reset_pt']= 1
                    elif source_reuse_mode == 'PT+D':
                        params['reset_pt']= 1
                        #dropout = 1
                        #dropout_rate = 0.5
                    elif source_reuse_mode == 'R':
                        params['reset_random']= 1
                        #dropout = 1
                        #dropout_rate = 0.5
                        
                    print 'switch datasets'
                    print target_dataset + ' is the source dataset'
                    tmp = target_dataset
                    target_dataset = source_dataset
                    source_dataset = tmp
                    print target_dataset + ' is the target dataset'
            
                    source_outputs_dir = target_outputs_dir
                    target_outputs_dir = '{0}{1}_reusing_{2}_TL{3}/'\
                         .format(results_dir,
                                 target_dataset,
                                 source_dataset,
                                 str(transfer))
                    print 'source_outputs_dir', source_outputs_dir
                    print 'target_outputs_dir', target_outputs_dir
                    
                    run_n_times(params, nr_reps,target_dataset,source_dataset,
                            training_data_fraction,source_outputs_dir,target_outputs_dir,source_reuse_mode, retrain_ft_layers,
                            approach, transfer)
                    
    
    
    
    elif approach == 'ETL':
        print 'approach is ETL'
        params['retrain'] = 0
        params['reset_pt'] = 0
        #print experiment
        
        if source_reuse_mode == None:
        
            source_outputs_dir = None
            target_outputs_dir = '{0}{1}_{2:4.2f}/'\
            .format(results_dir,
                    target_dataset,
                    training_data_fraction)        
        print target_outputs_dir
        run_n_times(params, nr_reps,target_dataset,source_dataset,
                training_data_fraction,source_outputs_dir,target_outputs_dir,source_reuse_mode, retrain_ft_layers,
                approach)    