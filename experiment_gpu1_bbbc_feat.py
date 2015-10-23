from approaches import *

results_dir = 'results/test_STS_full_spect/'
results_dir = 'results/test_MSTS[1111]/'
results_dir = 'results/test_bbbc_with_mnist/'
results_dir = 'results/BL_bbbc_feat3/'
results_dir = 'results/BL_bbbc_feat4/'
results_dir = 'results/TL_bbbc_moa2comp_feat4/'
results_dir = 'results/TL_bbbc_comp2moa_feat4/'
results_dir = 'results/BL_bbbc_20rep/'
results_dir = 'results/BL_bbbc_20rep_5layers/'
#results_dir = 'results/TL_bbbc_comp2moa_5layers/'

params = {
'finetune_lr':0.1,
'pretraining_epochs':50, #30,#60,#120,#80,#40,#15
'pretrain_lr':0.001,
'training_epochs': 1000, #2,#
'hidden_layers_sizes': [453,453*2,453*3,453*4, 453*5], #[453,453*2,453*3],  #[, # #[453,453*2], #[453],   #[453,453], #[24 * 24, 20 * 20, 16 * 16], #[1000,1000,1000], # [5,5],#[100,200,300], #
'batch_size':100,         
'output_fold':results_dir, #output_fold,
'rng_seed': 1234 #89677
}

nr_reps = 2 #20 #1 #5 #10


#training_data_fractions = [1.00,0.50,0.40,0.30,0.20,0.15,0.10,0.05]
training_data_fractions = [1.00, 0.50]
training_data_fractions = [1.00, 0.50,0.40,0.30,0.20,0.15,0.10,0.05]
training_data_fractions = [1.00]


#[approach, target_dataset, source_dataset, source_reuse_mode,dropout_weights,dropout_rate, retrain_ft_layers] = experiment
#experiments = [['BL'   , 'bbbc+feat3',                      None,     None    , None       ],]
#experiments = [['BL'   , 'bbbc+comp',                      None,     None    , None       ],]
#experiments = [['BL'   , 'bbbc+moa',                      None,     None    , None       ],]
#TL expt
#experiments = [['TL'   , 'bbbc+comp',   'bbbc+moa',  'PT+FT' , [1,1,1,1] ],]
# experiments = [['TL'   ,  'bbbc+comp',   'bbbc+moa',  'PT+FT' , [0,1,1,1] ],
#                ['TL'   ,  'bbbc+comp',   'bbbc+moa',  'PT+FT' , [0,0,1,1] ],
#                ['TL'   ,  'bbbc+comp',   'bbbc+moa',  'PT+FT' , [0,0,0,1] ],]

# Experiments for 20 repetitions:
# experiments = [['BL'   , 'bbbc+comp',      None,     None    , None       ],
#                ['BL'   ,  'bbbc+moa',       None,     None    , None      ],
#                ['TL'   ,  'bbbc+moa',   'bbbc+comp',  'PT+FT' , [1,1,1,1] ],
#                ['TL'   ,  'bbbc+moa',   'bbbc+comp',  'PT+FT' , [0,1,1,1] ],
#                ['TL'   ,  'bbbc+moa',   'bbbc+comp',  'PT+FT' , [0,0,1,1] ],
#                ['TL'   ,  'bbbc+moa',   'bbbc+comp',  'PT+FT' , [0,0,0,1] ],]




# Experiments for TL with 5 layers
experiments = [['BL'   , 'bbbc+comp',      None,     None    , None       ],]
#               ['BL'   ,  'bbbc+moa',       None,     None    , None      ],]
# experiments = [['TL'   ,  'bbbc+moa',   'bbbc+comp',  'PT+FT' , [1,1,1,1,1,1] ],
#                ['TL'   ,  'bbbc+moa',   'bbbc+comp',  'PT+FT' , [0,1,1,1,1,1] ],
#                ['TL'   ,  'bbbc+moa',   'bbbc+comp',  'PT+FT' , [0,0,1,1,1,1] ],
#                ['TL'   ,  'bbbc+moa',   'bbbc+comp',  'PT+FT' , [0,0,0,1,1,1] ],
#                ['TL'   ,  'bbbc+moa',   'bbbc+comp',  'PT+FT' , [0,0,0,0,1,1] ],
#                ['TL'   ,  'bbbc+moa',   'bbbc+comp',  'PT+FT' , [0,0,0,0,0,1] ],]



#experiments = [#['BL'   , 'mnist_64x80',   None,             None    , None       ],
               #['TL'   , 'bbbc',         'mnist_64x80',     'PT+FT' , [1,1,1,1]  ],]


#experiments = [['BL'   , 'bbbc',                      None,     None    , None       ],]

# experiments = [#['BL'   , 'mnist',                      None,     None    , None       ],           
#                ['MSTS' ,'chars74k_lowercase28x28',  'mnist',     'PT+FT' , [1,1,1,1]  ],]

# experiments = [['BL'  ,  'sh2_1cs_2p_3o',             None,      None    , None       ],
#                ['MSTS',  'sh1_1cs_2p_3o',  'sh2_1cs_2p_3o',      'PT+FT' , [0,0,1,1]  ],]

for experiment in experiments:
    print 'experiment=',experiment
    [approach, target_dataset, source_dataset, source_reuse_mode, retrain_ft_layers] = experiment
    for training_data_fraction in training_data_fractions:
        print 'training_data_fraction=',training_data_fraction
        bl_or_tl(experiment,training_data_fraction,results_dir, params,nr_reps)