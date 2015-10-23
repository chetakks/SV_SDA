from approaches import *

#results_dir = 'results/test_BBBC_MSTS[1111]/'
#results_dir = 'results/test_BBBC_MSTS[1111]com/'
#results_dir = 'results/test_BBBC_MSTS[0011]/'
results_dir = 'results/test_BBBC_MSTS[0011]moa+com/' # both moa, and com as target


params = {
'finetune_lr':0.1,
'pretraining_epochs':30,
'pretrain_lr':0.001,
'training_epochs': 1000, #2,#
'hidden_layers_sizes': [453,453*2,453*3],
'batch_size':100,         
'output_fold':results_dir, #output_fold,
'rng_seed': 1234
}

nr_reps = 20

training_data_fractions = [1.00]

experiments = [#['BL'   ,'bbbc+comp',     None,        None    , None       ],           
               #['MSTS' ,'bbbc+moa',  'bbbc+comp',     'PT+FT' , [1,1,1,1]  ],
               #['MSTS' ,'bbbc+comp',  'bbbc+moa',     'PT+FT' , [1,1,1,1]  ],
               ['MSTS' ,'bbbc+comp',  'bbbc+moa',     'PT+FT' , [0,0,1,1]  ],
               ['MSTS' ,'bbbc+moa',  'bbbc+comp',     'PT+FT' , [0,0,1,1]  ],
               ]


for experiment in experiments:
    print 'experiment=',experiment
    [approach, target_dataset, source_dataset, source_reuse_mode, retrain_ft_layers] = experiment
    for training_data_fraction in training_data_fractions:
        print 'training_data_fraction=',training_data_fraction
        bl_or_tl(experiment,training_data_fraction,results_dir, params,nr_reps)