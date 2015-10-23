from approaches import *
import numpy as np
from pprint import pprint
results_dir = 'Grid_results/'

params = {
'finetune_lr':0.01, #0.01,
'pretraining_epochs':50,
'pretrain_lr':0.001,
'training_epochs': 1000,
'hidden_layers_sizes': [1000,1000,1000],
'batch_size':1,         
'output_fold':results_dir, #output_fold,
'rng_seed': 1234
}

nr_reps = 1
training_data_fractions = [1.00]

# # Experiments with raw pixels
# pts =  [5, 10, 20, 30, 40, 50]
# nns =  [[3000,3000],[3000,3000,3000], [500, 500], [500,500,500], [2000, 2000]]
#  
# experiments = [['BL'   ,'csip_ar1',     None,        None    , None       ],
#                ['BL'   ,'csip_ar0',     None,        None    , None       ],
#                ['BL'   ,'csip_br0',     None,        None    , None       ],
#                ['BL'   ,'csip_br1',     None,        None    , None       ],
#                ['BL'   ,'csip_bf0',     None,        None    , None       ],
#                ['BL'   ,'csip_cf0',     None,        None    , None       ],
#                ['BL'   ,'csip_cr0',     None,        None    , None       ],
#                ['BL'   ,'csip_cr1',     None,        None    , None       ],
#                ['BL'   ,'csip_df0',     None,        None    , None       ],
#                ['BL'   ,'csip_dr0',     None,        None    , None       ],
#                ]
#        
# for experiment in experiments:
#     [approach, target_dataset, source_dataset, source_reuse_mode, retrain_ft_layers] = experiment
#     for id_pt, pt in enumerate(pts):
#         params['pretraining_epochs'] = pt
#         for idx, nn in enumerate(nns):
#             params['hidden_layers_sizes'] = nn
#             retrain_ft_layers = None
#             #retrain_ft_layers = retrain_layers[idx]
#             #tranferred_layers = tranfer_layers[idx]
#             experiment = [approach, target_dataset, source_dataset, source_reuse_mode, retrain_ft_layers]
#             for training_data_fraction in training_data_fractions:
#                 print 'experiment=',experiment
#                 print 'params='
#                 pprint(params, width=1)
#                 print 'training_data_fraction=',training_data_fraction
#                 bl_or_tl(experiment,training_data_fraction,results_dir, params,nr_reps)
                
#############################################################################################################
#############################################################################################################

# # Experiments with super vectors
# results_dir = 'Grid_results_sv/'
# pts =  [5, 10, 20, 30, 40, 50]
# nns =  [[500, 500],[500,500,500],[1000,1000],[1000,1000,1000],[2000,2000],[2000,2000,2000]]
# 
# experiments = [['BL'   ,'csip_sv_ar1',     None,        None    , None       ],
#                ['BL'   ,'csip_sv_ar0',     None,        None    , None       ],
#                ['BL'   ,'csip_sv_br0',     None,        None    , None       ],
#                ['BL'   ,'csip_sv_br1',     None,        None    , None       ],
#                ['BL'   ,'csip_sv_bf0',     None,        None    , None       ],
#                ['BL'   ,'csip_sv_cf0',     None,        None    , None       ],
#                ['BL'   ,'csip_sv_cr0',     None,        None    , None       ],
#                ['BL'   ,'csip_sv_cr1',     None,        None    , None       ],
#                ['BL'   ,'csip_sv_df0',     None,        None    , None       ],
#                ['BL'   ,'csip_sv_dr0',     None,        None    , None       ],
#                ]

# # Experiments with super vectors with fix scale
# results_dir = 'Grid_results_sv_fix/'
# pts =  [5, 10, 20, 30, 40, 50]
# pts =  [ 40, 50]
# nns =  [[500, 500],[500,500,500],[1000,1000],[1000,1000,1000],[2000,2000],[2000,2000,2000]]
# nns =  [[3000,3000,3000]]
# experiments = [#['BL'   ,'csip_sv_fix_ar1',     None,        None    , None       ],
#                #['BL'   ,'csip_sv_fix_ar0',     None,        None    , None       ],
#                #['BL'   ,'csip_sv_fix_br0',     None,        None    , None       ],
#                #['BL'   ,'csip_sv_fix_br1',     None,        None    , None       ],
#                #['BL'   ,'csip_sv_fix_bf0',     None,        None    , None       ],
#                ['BL'   ,'csip_sv_fix_cf0',     None,        None    , None       ],
#                ['BL'   ,'csip_sv_fix_cr0',     None,        None    , None       ],
#                ['BL'   ,'csip_sv_fix_cr1',     None,        None    , None       ],
#                ['BL'   ,'csip_sv_fix_df0',     None,        None    , None       ],
#                ['BL'   ,'csip_sv_fix_dr0',     None,        None    , None       ],
#                ]

# # Experiments with super vectors with fix scale
# results_dir = 'Grid_results_sv_fix_128/'
# pts =  [5, 10, 20, 30, 40, 50]
# pts =  [1]
# nns =  [[500, 500],[500,500,500],[1000,1000],[1000,1000,1000],[2000,2000],[2000,2000,2000]]
# nns =  [[1000,1000]]
# nns =  [[2,2], [5,5], [10,10],[20,20] ]
# experiments = [#['BL'   ,'csip_sv_fix_128_ar1',     None,        None    , None       ],
#                #['BL'   ,'csip_sv_fix_128_ar0',     None,        None    , None       ],
#                #['BL'   ,'csip_sv_fix_128_br0',     None,        None    , None       ],
#                #['BL'   ,'csip_sv_fix_128_br1',     None,        None    , None       ],
#                #['BL'   ,'csip_sv_fix_128_bf0',     None,        None    , None       ],
#                #['BL'   ,'csip_sv_fix_128_cf0',     None,        None    , None       ],
#                #['BL'   ,'csip_sv_fix_128_cr0',     None,        None    , None       ],
#                ['BL'   ,'csip_sv_fix_128_cr1',     None,        None    , None       ],
#                ['BL'   ,'csip_sv_fix_128_df0',     None,        None    , None       ],
#                ['BL'   ,'csip_sv_fix_128_dr0',     None,        None    , None       ],
#                ]
      
      
# # Experiments with super vectors with fix scale
# results_dir = 'Grid_results_sv_fix_2/'
# pts =  [5, 10, 20, 30, 40, 50]
# #pts =  [1]
# nns =  [[500, 500],[500,500,500],[1000,1000],[1000,1000,1000],[2000,2000],[2000,2000,2000]]
# nns =  [[1000,1000]]
# #nns =  [[2,2], [5,5], [10,10],[20,20] ]
# experiments = [['BL'   ,'csip_sv_fix_2_ar1',     None,        None    , None       ],
#                #['BL'   ,'csip_sv_fix_2_ar0',     None,        None    , None       ],
#                #['BL'   ,'csip_sv_fix_2_br0',     None,        None    , None       ],
#                #['BL'   ,'csip_sv_fix_2_br1',     None,        None    , None       ],
#                #['BL'   ,'csip_sv_fix_2_bf0',     None,        None    , None       ],
#                #['BL'   ,'csip_sv_fix_2_cf0',     None,        None    , None       ],
#                #['BL'   ,'csip_sv_fix_2_cr0',     None,        None    , None       ],
#                ['BL'   ,'csip_sv_fix_2_cr1',     None,        None    , None       ],
#                ['BL'   ,'csip_sv_fix_2_df0',     None,        None    , None       ],
#                ['BL'   ,'csip_sv_fix_2_dr0',     None,        None    , None       ],
#                ]      

# # Experiments with super vectors with domain specialization, (or)l2 norm, (or) raw domain specialization
# results_dir = 'Grid_results_ds/'
# #pts =  [20, 10, 20, 30, 40, 50]
# pts =  [25]
# #nns =  [[500, 500],[500,500,500],[1000,1000],[1000,1000,1000],[2000,2000],[2000,2000,2000]]
# nns =  [[1000,1000]]
# #nns =  [[2,2], [5,5], [10,10],[20,20] ]
# experiments = [#['BL'   ,'csip_sv_ds_ar1',     None,        None    , None       ],
#                #['BL'   ,'csip_sv_l2_ar1',     None,        None    , None       ],
#                ['BL'   ,'csip_ar1',          None,        None    , None       ],
#                ['TL'   ,'csip_ar0',    'csip_ar1',        'PT'    , [1,1,1]    ],
#                ['TL'   ,'csip_br0',    'csip_ar1',        'PT'    , [1,1,1]    ],
#                ['TL'   ,'csip_br1',    'csip_ar1',        'PT'    , [1,1,1]    ],
#                ['TL'   ,'csip_bf0',    'csip_ar1',        'PT'    , [1,1,1]    ],
#                ['TL'   ,'csip_cf0',    'csip_ar1',        'PT'    , [1,1,1]    ],
#                ['TL'   ,'csip_cr0',    'csip_ar1',        'PT'    , [1,1,1]    ],
#                ['TL'   ,'csip_cr1',    'csip_ar1',        'PT'    , [1,1,1]    ],
#                ['TL'   ,'csip_df0',    'csip_ar1',        'PT'    , [1,1,1]    ],
#                ['TL'   ,'csip_dr0',    'csip_ar1',        'PT'    , [1,1,1]    ],
#                #['BL'   ,'csip_sv_ar1',     None,        None    , None       ],
#                ] 

# # supervector domina specialization
# results_dir = 'Grid_results_sv_ds/'
# #pts =  [20, 10, 20, 30, 40, 50]
# pts =  [25]
# #nns =  [[500, 500],[500,500,500],[1000,1000],[1000,1000,1000],[2000,2000],[2000,2000,2000]]
# nns =  [[1000,1000]]
# #nns =  [[2,2], [5,5], [10,10],[20,20] ]
# experiments = [#['BL'   ,'csip_sv_ds_ar1',     None,        None    , None       ],
#                #['BL'   ,'csip_sv_l2_ar1',     None,        None    , None       ],
#                ['BL'   ,'csip_sv_ar1',          None,        None    , None       ],
#                ['TL'   ,'csip_sv_ar0',    'csip_sv_ar1',        'PT'    , [1,1,1]    ],
#                ['TL'   ,'csip_sv_br0',    'csip_sv_ar1',        'PT'    , [1,1,1]    ],
#                ['TL'   ,'csip_sv_br1',    'csip_sv_ar1',        'PT'    , [1,1,1]    ],
#                ['TL'   ,'csip_sv_bf0',    'csip_sv_ar1',        'PT'    , [1,1,1]    ],
#                ['TL'   ,'csip_sv_cf0',    'csip_sv_ar1',        'PT'    , [1,1,1]    ],
#                ['TL'   ,'csip_sv_cr0',    'csip_sv_ar1',        'PT'    , [1,1,1]    ],
#                ['TL'   ,'csip_sv_cr1',    'csip_sv_ar1',        'PT'    , [1,1,1]    ],
#                ['TL'   ,'csip_sv_df0',    'csip_sv_ar1',        'PT'    , [1,1,1]    ],
#                ['TL'   ,'csip_sv_dr0',    'csip_sv_ar1',        'PT'    , [1,1,1]    ],
#                #['BL'   ,'csip_sv_ar1',     None,        None    , None       ],
#                ] 

# supervector domina specialization
results_dir = 'Grid_results_normalized/'
#pts =  [20, 10, 20, 30, 40, 50]
pts =  [25]

#nns =  [[500, 500],[500,500,500],[1000,1000],[1000,1000,1000],[2000,2000],[2000,2000,2000]]
nns =  [[1000,1000]]
#nns =  [[2,2], [5,5], [10,10],[20,20] ]
experiments = [#['BL'   ,'csip_sv_ds_ar1',     None,        None    , None       ],
               #['BL'   ,'csip_sv_l2_ar1',     None,        None    , None       ],
               #['BL'   ,'csip_sv_ar1',     None,        None    , None       ],
               ['BL'   ,'csip_noar1',          None,        None    , None       ],
             
               ] 

for experiment in experiments:
    [approach, target_dataset, source_dataset, source_reuse_mode, retrain_ft_layers] = experiment
    for id_pt, pt in enumerate(pts):
        params['pretraining_epochs'] = pt
        for idx, nn in enumerate(nns):
            params['hidden_layers_sizes'] = nn
            #retrain_ft_layers = None
            #retrain_ft_layers = retrain_layers[idx]
            #tranferred_layers = tranfer_layers[idx]
            experiment = [approach, target_dataset, source_dataset, source_reuse_mode, retrain_ft_layers]
            for training_data_fraction in training_data_fractions:
                print 'experiment=',experiment
                print 'params='
                pprint(params, width=1)
                print 'training_data_fraction=',training_data_fraction
                bl_or_tl(experiment,training_data_fraction,results_dir, params,nr_reps)
                
                
#############################################################################################################                
                
# nns =                     [[3000, 3000, 3000] , [3000, 3000],[3000, 3000, 3000, 3000] , [500, 500] , [2000, 2000] ]

#experiments = [['BL'   ,'csip_ar1',     None,        None    , None       ],]
#experiments = [['BL'   ,'csip_sv_ar1',     None,        None    , None       ],]
#experiments = [['BL'   ,'csip_sv_ar1',     None,        None    , retrain_layers  ],]
#experiments = [['BL'   ,'csip_dct_ar1',     None,        None    , None       ],]
#experiments = [['BL'   ,'csip_sv_dct_ar1',     None,        None    , None       ],]
    

#############################################################################################################
# experiments = [['BL'   ,'sh1_1cs_2p_3o',     None,        None    , None       ],
#                ['BL'   ,'sh2_1cs_2p_3o',     None,        None    , None       ],
#                ['BL'   ,'sh102_1cs_2p_3o',   None,        None    , None       ],
#                ['BL'   ,'sh202_1cs_2p_3o',   None,        None    , None       ],
#                ['TL'   ,'sh1_1cs_2p_3o', 'sh2_1cs_2p_3o', 'PT+FT' , [1,1,1,1]  ],
#                ['TL',   'sh2_1cs_2p_3o', 'sh1_1cs_2p_3o', 'PT+FT' , [1,1,1,1]  ], 
#                ['STS',  'sh1_1cs_2p_3o', ['TL','sh2_1cs_2p_3o', 'sh1_1cs_2p_3o',  'PT+FT' , [1,1,1,1]  ],  
#                                                                                   'PT+FT' , [1,1,1,1]  ],  
#                ['MSTS' ,'sh1_1cs_2p_3o',  'sh2_1cs_2p_3o',     'PT+FT' , [1,1,1,1]  ],
#                ]




# for experiment in experiments:
#     print 'experiment=',experiment
#     [approach, target_dataset, source_dataset, source_reuse_mode, retrain_ft_layers] = experiment
#     for training_data_fraction in training_data_fractions:
#         print 'training_data_fraction=',training_data_fraction
#         bl_or_tl(experiment,training_data_fraction,results_dir, params,nr_reps)
        