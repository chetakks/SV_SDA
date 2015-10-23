from approaches import *

results_dir = 'results/test_fprint/'

params = {
'finetune_lr':0.1,
'pretraining_epochs':20, #30,#60,#120,#80,#40,#15
'pretrain_lr':0.001,
'training_epochs': 1000, #2,#
'hidden_layers_sizes': [100,100,100,100], #[453,453*2,453*3],  #[, # #[453,453*2], #[453],   #[453,453], #[24 * 24, 20 * 20, 16 * 16], #[1000,1000,1000], # [5,5],#[100,200,300], #
'batch_size':100,         
'output_fold':results_dir, #output_fold,
'rng_seed': 1234 #89677
}

nr_reps = 2 #20 #1 #5 #10


#training_data_fractions = [1.00,0.50,0.40,0.30,0.20,0.15,0.10,0.05]
training_data_fractions = [1.00, 0.50]
training_data_fractions = [1.00, 0.50,0.40,0.30,0.20,0.15,0.10,0.05]
training_data_fractions = [1.00]

experiments = [['BL'   , 'fprint',      None,     None    , None       ],]

for experiment in experiments:
    print 'experiment=',experiment
    [approach, target_dataset, source_dataset, source_reuse_mode, retrain_ft_layers] = experiment
    for training_data_fraction in training_data_fractions:
        print 'training_data_fraction=',training_data_fraction
        bl_or_tl(experiment,training_data_fraction,results_dir, params,nr_reps)