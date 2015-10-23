from approaches import *

results_dir = 'results/test_STS_full_spect/'
results_dir = 'results/test_MSTS[1111]/'

#======================================================#
# Experimental setup for MSTS approach with cyclic STS 
#======================================================#

# params = {
# 'finetune_lr':0.1,
# 'pretraining_epochs':40,#60,#120,#80,#40,#15
# 'pretrain_lr':0.001,
# 'training_epochs': 1000, #2,#
# 'hidden_layers_sizes': [24 * 24, 20 * 20, 16 * 16], #[1000,1000,1000], # [5,5],#[100,200,300], #
# 'batch_size':1,         
# 'output_fold':results_dir, #output_fold,
# 'rng_seed': 1234 #89677
# }
# 
# nr_reps = 10 #5 #10
# 
# 
# #training_data_fractions = [1.00,0.50,0.40,0.30,0.20,0.15,0.10,0.05]
# training_data_fractions = [1.00, 0.50]
# training_data_fractions = [1.00, 0.50,0.40,0.30,0.20,0.15,0.10,0.05]
# training_data_fractions = [1.00]
# 
# 
# #[approach, target_dataset, source_dataset, source_reuse_mode,dropout_weights,dropout_rate, retrain_ft_layers] = experiment
# 
# experiments = [['BL'   ,'chars74k_lowercase28x28',             None,     None    , None       ],           
#                ['MSTS' , 'mnist',         'chars74k_lowercase28x28',     'PT+FT' , [1,1,1,1]  ],]
# 
# # experiments = [['BL'  ,  'sh2_1cs_2p_3o',             None,      None    , None       ],
# #                ['MSTS',  'sh1_1cs_2p_3o',  'sh2_1cs_2p_3o',      'PT+FT' , [0,0,1,1]  ],]


#======================================================#
# Experimental setup for STS approach with MNIST
#======================================================#

results_dir = 'results/test_STS_full_drp/'
results_dir = '/home/aditya/store/Theano/STS_full/results/test_STS_full_drp/'

params = {
    'finetune_lr':0.1,
    'pretraining_epochs':50,#120,#80,#40,#15
    'pretrain_lr':0.001,
    'training_epochs':1000, #2,#
    'hidden_layers_sizes': [24 * 24, 20 * 20, 16 * 16], #[1000,1000,1000], #
    'batch_size':1,         
    'output_fold':results_dir, #output_fold,
    'rng_seed':89677
    }

nr_reps = 10
training_data_fractions = [1.00, 0.50,0.40,0.30,0.20,0.15,0.10,0.05]
#training_data_fractions = [1.00]

experiments =     [#['BL' , 'mnist',                          None,                              None    , None       ],
                   #['TL' , 'chars74k_uppercase28x28',     'mnist',                                  'R' , [1,1,0,1]  ],
                   #['TL' , 'chars74k_uppercase28x28',     'mnist',                                'R+D' , [1,1,0,1]  ],
                   #['TL' , 'chars74k_uppercase28x28',     'mnist',                                 'PT' , [1,1,0,1]  ],
                   
                   # above experiments completed
                   
                   #['TL' , 'chars74k_uppercase28x28',     'mnist',                               'PT+D' , [1,1,0,1]  ],
                   
                   # above experiment stopped at 6th Repetition.
                   
                   #['TL' , 'chars74k_uppercase28x28',     'mnist',                              'PT+FT' , [1,1,0,1]  ],
                   #['TL' , 'chars74k_uppercase28x28',     'mnist',                            'PT+FT+D' , [1,1,0,1]  ],
                   #['STS', 'mnist',['TL','chars74k_uppercase28x28','mnist','PT+FT',[1,1,0,1] ], 'PT+FT' , [1,1,0,1]  ],
                   
                    # above experiment stopped at 8th Repetition.
                   
                    #['BL' , 'chars74k_uppercase28x28',        None,                              None    , None       ],
                    #['TL' , 'mnist',   'chars74k_uppercase28x28',                                'PT+FT' , [1,1,0,1]  ],
                    #['STS', 'chars74k_uppercase28x28',     ['TL','mnist','chars74k_uppercase28x28','PT+FT', [1,1,0,1] ],
                    #                                                                              'PT+FT' , [1,1,0,1] ],]

######Second GPU
                    ['BL' , 'chars74k_lowercase28x28',          None,                              None    , None     ],
                    ['TL' , 'chars74k_lowercase28x28',     'mnist',                                  'R' , [1,1,0,1]  ],
                    ['TL' , 'chars74k_lowercase28x28',     'mnist',                                'R+D' , [1,1,0,1]  ],
                    ['TL' , 'chars74k_lowercase28x28',     'mnist',                                 'PT' , [1,1,0,1]  ],
                    ['TL' , 'chars74k_lowercase28x28',     'mnist',                               'PT+D' , [1,1,0,1]  ],
                    ['TL' , 'chars74k_lowercase28x28',     'mnist',                              'PT+FT' , [1,1,0,1]  ],
                    ['TL' , 'chars74k_lowercase28x28',     'mnist',                            'PT+FT+D' , [1,1,0,1]  ],
                    #['STS', 'mnist',['TL','chars74k_lowercase28x28','mnist','PT+FT',[1,1,0,1] ], 'PT+FT' , [1,1,0,1]  ],
                    ['TL' , 'mnist',   'chars74k_lowercase28x28',                                'PT+FT' , [1,1,0,1]  ],
                    ['STS', 'chars74k_lowercase28x28',     ['TL','mnist','chars74k_lowercase28x28','PT+FT', [1,1,0,1] ],
                                                                                                  'PT+FT' , [1,1,0,1] ],]
###############
    
#                     ['TL' ,  'mnist',     'chars74k_uppercase28x28',                                  'R' , [1,1,0,1] ],
#                     ['TL' ,  'mnist',     'chars74k_uppercase28x28',                                'R+D' , [1,1,0,1] ],
#                     ['TL' ,  'mnist',     'chars74k_uppercase28x28',                                 'PT' , [1,1,0,1] ],
#                     ['TL' ,  'mnist',     'chars74k_uppercase28x28',                               'PT+D' , [1,1,0,1] ],
#                     ['TL' ,  'mnist',     'chars74k_uppercase28x28',                              'PT+FT' , [1,1,0,1] ],
#                     ['TL' ,  'mnist',     'chars74k_uppercase28x28',                            'PT+FT+D' , [1,1,0,1] ],
#                    ['STS', 'chars74k_uppercase28x28',   ['TL','mnist','chars74k_uppercase28x28','PT+FT+D', [1,1,0,1] ],
#                                                                                                'PT+FT+D' , [1,1,0,1] ],]




for experiment in experiments:
    print 'experiment=',experiment
    [approach, target_dataset, source_dataset, source_reuse_mode, retrain_ft_layers] = experiment
    for training_data_fraction in training_data_fractions:
        print 'training_data_fraction=',training_data_fraction
        bl_or_tl(experiment,training_data_fraction,results_dir, params,nr_reps)