from approaches import *

#results_dir = 'results/test_BBBC_MSTS[1111]/'
#results_dir = 'results/test_BBBC_MSTS[1111]com/'
#results_dir = 'results/test_BBBC_MSTS[0011]/'
#results_dir = 'results/test_BBBC_MSTS[0011]moa+com/' # both moa, and com as target
results_dir = 'results/test_BBBC_ETL/'

params = {
'finetune_lr':0.1,
'pretraining_epochs':30, #30,
'pretrain_lr':0.001,
'training_epochs': 30,# 1000, #
'hidden_layers_sizes': [30,30,30], #[453,453*2,453*3],
'batch_size':100,         
'output_fold':results_dir, #output_fold,
'rng_seed': 1234
}

nr_reps = 10# 20

training_data_fractions = [0.05]

experiments = [['ETL'   ,'bbbc+moa',     None,        None    , None       ],           
               ]


for experiment in experiments:
    print 'experiment=',experiment
    [approach, target_dataset, source_dataset, source_reuse_mode, retrain_ft_layers] = experiment
    for training_data_fraction in training_data_fractions:
        print 'training_data_fraction=',training_data_fraction
        bl_or_tl(experiment,training_data_fraction,results_dir, params,nr_reps)