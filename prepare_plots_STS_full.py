# NOTE: If openblas is installed (libopenblas-base and libopenblas-dev in
# Ubuntu) and multi-processing is needed (enabled by setting nr_processes>0),
# then the OMP_NUM_THREADS environment variable should be set to 1.
    
#from data_handling import *
#from execution import *
from plotting import *

import numpy
import string

if __name__ == "__main__":
    
    results_dir = 'results/test_STS/'
    

    params = {
    'finetune_lr':0.1,
    'pretraining_epochs':120,#80,#40,#15
    'pretrain_lr':0.001,
    'training_epochs':1000, #2,#
    'n_ins': 28 * 28, #64*48, #
    'hidden_layers_sizes': [24 * 24, 20 * 20, 16 * 16], #[1000,1000,1000], #
    'n_outs':3,#26
    'n_outs_b':3,#26
    'batch_size':1,         
    'output_fold':results_dir, #output_fold,
    'rng_seed':89677
    }
    
    nr_reps = 3#20 #5 #10
    repeat = nr_reps
    
    #training_data_fractions = [1.00,0.50,0.40,0.30,0.20,0.15,0.10,0.05]
    training_data_fractions = [0.50,1.00]
    #training_data_fractions = [0.05,0.10,0.15,0.20,0.30,0.40,0.50,1.00]

    
    #[approach, target_dataset, source_dataset, source_reuse_mode, retrain_ft_layers] = experiment
    #experiments = [['ordinal'    ,None         ,None   , 1,'$ordinal$'                             ,'-' ,2,'k'],]
    
    experiments = [['BL' , 'Pc',     None,                                      None    , None       ],
                   ['TL' , 'Pn',     'Pc',                                      'PT+FT' , [0,0,1,1]  ],
                   ['STS', 'Pc',    ['TL' , 'Pn', 'Pc', 'PT+FT' , [0,0,1,1]  ], 'PT+FT' , [0,0,1,1]  ],]
    
   # experiments = [['BL', 'Pc',  None,  None    , None      ],]
                   #['BL', 'Pn',  None,  None    , None      ],
                   #['TL', 'Pc',  'Pn',  'PT'    , [0,0,1,1,1,1,1,1]   ],
                   #['TL', 'Pc',  'Pn',  'PT+FT' , [0,0,1,1,1,1,1,1]   ],]
                   
    #experiments = [['Pc'    ,None         ,None   , 1,'$Pc_{ds.ori}$'                             ,'-' ,2,'k'],]
    #experiments = [['pc'    ,None         ,None   , 1,'$pc_{ds.ori}$'                             ,'-' ,2,'k'],]
    #experiments = [['pc'    ,'Pn'         ,'PT+FT' , [0,0,1,1,1,1,1,1],'$pc_{ds.ori}$'             ,'-' ,2,'k'],]
    #experiments = [[ 'Pn',  'pc',  'PT+FT' , [0,0,1,1,1,1,1,1],'$Pn_{ds.ori}$'             ,'-' ,2,'k'],]  
    
    #experiments = [['ordinal'    ,None         ,None   , 1,'$ordinal$'                             ,'-' ,2,'k'],]
    
    n_ds_max = 5000
    xmax = 6000  # 900#
    c = 3  #5
    
#     #change in distribution experiments with irregular sizes 
    results_dir = 'results/ICMLA/SA_CD_shapes/'   
#     experiments = [[ 'Pn',  None,  None, None,'$Pn_{ds.ori}$'             ,'-' ,2,'k'],]    
#     ylim = None
#     file_name = 'Pn.png'
#     plot_e2(results_dir,experiments,training_data_fractions,n_ds_max,xmax,c,nr_reps,file_name,ylim=ylim)
    
#     experiments = [[ 'Pc',  None,  None, None,'$Pc_{ds.ori}$'             ,'-' ,2,'k'],]    
#     ylim = None
#     file_name = 'Pc.png'
#     plot_e2(results_dir,experiments,training_data_fractions,n_ds_max,xmax,c,nr_reps,file_name,ylim=ylim)
#     
#     experiments = [[ 'Pc',  'Pn',  'PT+FT' , [1,1,0,0,0,0,0,0],'$Pc_reusing_Pn$'             ,'-' ,2,'k'],]    
#     ylim = None
#     file_name = 'Pc_reusing_Pn1.png'
#     plot_e2(results_dir,experiments,training_data_fractions,n_ds_max,xmax,c,nr_reps,file_name,ylim=ylim)
#     
#     experiments = [[ 'Pc',  'Pn',  'PT+FT' , [1,1,1,1,0,0,0,0],'$Pc_reusing_Pn$'             ,'-' ,2,'k'],]    
#     ylim = None
#     file_name = 'Pc_reusing_Pn2.png'
#     plot_e2(results_dir,experiments,training_data_fractions,n_ds_max,xmax,c,nr_reps,file_name,ylim=ylim)
#     
#     experiments = [[ 'Pc',  'Pn',  'PT+FT' , [1,1,1,1,1,1,0,0],'$Pc_reusing_Pn$'             ,'-' ,2,'k'],]    
#     ylim = None
#     file_name = 'Pc_reusing_Pn3.png'
#     plot_e2(results_dir,experiments,training_data_fractions,n_ds_max,xmax,c,nr_reps,file_name,ylim=ylim)
#     
#     experiments = [[ 'Pc',  'Pn',  'PT+FT' , [1,1,1,1,1,1,1,1],'$Pc_reusing_Pn$'             ,'-' ,2,'k'],]    
#     ylim = None
#     file_name = 'Pc_reusing_Pn4.png'
#     plot_e2(results_dir,experiments,training_data_fractions,n_ds_max,xmax,c,nr_reps,file_name,ylim=ylim)


     
#     experiments = [[ 'Pc',  'Pn',  'PT+FT' , [0,0,0,0,0,0,1,1],'$Pc_reusing_Pn$'             ,'-' ,2,'k'],]    
#     ylim = None
#     file_name = 'Pc_reusing_Pn5.png'
#     plot_e2(results_dir,experiments,training_data_fractions,n_ds_max,xmax,c,nr_reps,file_name,ylim=ylim)
#      
#     experiments = [[ 'Pc',  'Pn',  'PT+FT' , [0,0,0,0,1,1,1,1],'$Pc_reusing_Pn$'             ,'-' ,2,'k'],]    
#     ylim = None
#     file_name = 'Pc_reusing_Pn6.png'
#     plot_e2(results_dir,experiments,training_data_fractions,n_ds_max,xmax,c,nr_reps,file_name,ylim=ylim)
#      
#     experiments = [[ 'Pc',  'Pn',  'PT+FT' , [0,0,1,1,1,1,1,1],'$Pc_reusing_Pn$'             ,'-' ,2,'k'],]    
#     ylim = None
#     file_name = 'Pc_reusing_Pn7.png'
#     plot_e2(results_dir,experiments,training_data_fractions,n_ds_max,xmax,c,nr_reps,file_name,ylim=ylim)

    experiments = [[ 'Pc',  None,  None, None,'$Pc_{ds.ori}$'             ,'-' ,2,'k'],
                   [ 'Pc',  'Pn',  'PT+FT' , [0,0,0,0,0,0,1,1],'$Pc_reusing_Pn$'             ,'--' ,2,'r'],  
                   [ 'Pc',  'Pn',  'PT+FT' , [0,0,0,0,1,1,1,1],'$Pc_reusing_Pn$'             ,':' ,2,'g'],
                   [ 'Pc',  'Pn',  'PT+FT' , [0,0,1,1,1,1,1,1],'$Pc_reusing_Pn$'             ,':' ,2,'y'],
                   [ 'Pc',  'Pn',  'PT+FT' , [1,1,1,1,1,1,1,1],'$Pc_reusing_Pn$'             ,':' ,2,'b'],]
    ylim = None
    file_name = 'Pc_CD.png'
    plot_e2(results_dir,experiments,training_data_fractions,n_ds_max,xmax,c,nr_reps,file_name,ylim=ylim)
    file_name = 'Pc_CD_times.png'
    plot_t3(results_dir,experiments,training_data_fractions,n_ds_max,xmax,c,nr_reps,file_name,ylim=ylim)

     



#     #change in distribution experiments with regular shapes 
#     results_dir = 'results/ICMLA/SA_CD_regular_shapes/'
#     experiments = [[ 'Pc_reg',  None,  None, None,'$Pc_reg_{ds.ori}$'             ,'-' ,2,'k'],]    
#     ylim = None
#     file_name = 'Pc_reg.png'
#     plot_e2(results_dir,experiments,training_data_fractions,n_ds_max,xmax,c,nr_reps,file_name,ylim=ylim)
#     
#     experiments = [[ 'Pn_reg',  None,  None, None,'$Pn_reg_{ds.ori}$'             ,'-' ,2,'k'],]    
#     ylim = None
#     file_name = 'Pn_reg.png'
#     plot_e2(results_dir,experiments,training_data_fractions,n_ds_max,xmax,c,nr_reps,file_name,ylim=ylim)
#      
#     experiments = [[ 'Pc_reg',  'Pn_reg',  'PT+FT' , [0,0,0,0,1,1,1,1] ,'$Pc_reg_reusing_Pn_reg$'             ,'-' ,2,'k'],]    
#     ylim = None
#     file_name = 'Pc_reg_reusing_Pn_reg1.png'
#     plot_e2(results_dir,experiments,training_data_fractions,n_ds_max,xmax,c,nr_reps,file_name,ylim=ylim)
#      
#     experiments = [[ 'Pc_reg',  'Pn_reg',  'PT+FT' , [1,1,1,1,0,0,0,0],'$Pc_reg_reusing_Pn_reg$'             ,'-' ,2,'k'],]    
#     ylim = None
#     file_name = 'Pc_reg_reusing_Pn_reg2.png'
#     plot_e2(results_dir,experiments,training_data_fractions,n_ds_max,xmax,c,nr_reps,file_name,ylim=ylim)
#      
#     experiments = [[ 'Pc_reg',  'Pn_reg',  'PT+FT' , [1,1,1,1,1,1,1,1],'$Pc_reg_reusing_Pn_reg$'             ,'-' ,2,'k'],]    
#     ylim = None
#     file_name = 'Pc_reg_reusing_Pn_reg3.png'
#     plot_e2(results_dir,experiments,training_data_fractions,n_ds_max,xmax,c,nr_reps,file_name,ylim=ylim)
#      
#     experiments = [[ 'Pn_reg',  'Pc_reg',  'PT+FT' , [0,0,0,0,1,1,1,1],'$Pn_reg_reusing_Pc_reg$'             ,'-' ,2,'k'],]    
#     ylim = None
#     file_name = 'Pn_reg_reusing_Pc_reg4.png'
#     plot_e2(results_dir,experiments,training_data_fractions,n_ds_max,xmax,c,nr_reps,file_name,ylim=ylim)
#       
#     experiments = [[ 'Pn_reg',  'Pc_reg',  'PT+FT' , [1,1,1,1,0,0,0,0],'$Pn_reg_reusing_Pc_reg$'             ,'-' ,2,'k'],]    
#     ylim = None
#     file_name = 'Pn_reg_reusing_Pc_reg6.png'
#     plot_e2(results_dir,experiments,training_data_fractions,n_ds_max,xmax,c,nr_reps,file_name,ylim=ylim)
#       
#     experiments = [[ 'Pn_reg',  'Pc_reg',  'PT+FT' , [1,1,1,1,1,1,1,1],'$Pn_reg_reusing_Pc_reg$'             ,'-' ,2,'k'],]    
#     ylim = None
#     file_name = 'Pn_reg_reusing_Pc_reg7.png'
#     plot_e2(results_dir,experiments,training_data_fractions,n_ds_max,xmax,c,nr_reps,file_name,ylim=ylim)
     
     
     
#     #change in set of class labels experiments with irregular shapes  
    #results_dir = 'results/ICMLA/SA_CL_shapes/'
    
    
#     experiments = [[ 'Pc' ,  None,    None,              None,'$Pc,\Omega_{03}$'             ,'-' ,2,'k'],
#                    [ 'Pc' ,'PcO2', 'PT+FT', [0,0,0,0,0,0,1,1],'$Pc,\Omega_{03}$ reusing $Pc,\Omega_{02}$ [0,0,0,1]' ,'--',2,'r'],
#                    [ 'Pc' ,'PcO2', 'PT+FT', [0,0,0,0,1,1,1,1],'$Pc,\Omega_{03}$ reusing $Pc,\Omega_{02}$ [0,0,1,1]' ,':' ,2,'g'],
#                    [ 'Pc' ,'PcO2', 'PT+FT', [0,0,1,1,1,1,1,1],'$Pc,\Omega_{03}$ reusing $Pc,\Omega_{02}$ [0,1,1,1]' ,':' ,2,'y'],
#                    [ 'Pc' ,'PcO2', 'PT+FT', [1,1,1,1,1,1,1,1],'$Pc,\Omega_{03}$ reusing $Pc,\Omega_{02}$ [1,1,1,1]' ,':' ,2,'b'],] 
#     ylim = None
#     file_name = 'Pc_CL.png'
#     plot_e3(results_dir,experiments,training_data_fractions,n_ds_max,xmax,c,nr_reps,file_name,ylim=ylim)
#     file_name = 'Pc_CL_times.png'
#     plot_t3(results_dir,experiments,training_data_fractions,n_ds_max,xmax,c,nr_reps,file_name,ylim=ylim)
#     
#     experiments = [[ 'PcO2' ,None,    None,              None,'$Pc,\Omega_{02}$'             ,'-' ,2,'k'],
#                    [ 'PcO2' ,'Pc', 'PT+FT', [0,0,0,0,0,0,1,1],'$Pc,\Omega_{02}$ reusing $Pc,\Omega_{03}$ [0,0,0,1]' ,'--',2,'r'],
#                    [ 'PcO2' ,'Pc', 'PT+FT', [0,0,0,0,1,1,1,1],'$Pc,\Omega_{02}$ reusing $Pc,\Omega_{03}$ [0,0,1,1]' ,':' ,2,'g'],
#                    [ 'PcO2' ,'Pc', 'PT+FT', [0,0,1,1,1,1,1,1],'$Pc,\Omega_{02}$ reusing $Pc,\Omega_{03}$ [0,1,1,1]' ,':' ,2,'y'],
#                    [ 'PcO2' ,'Pc', 'PT+FT', [1,1,1,1,1,1,1,1],'$Pc,\Omega_{02}$ reusing $Pc,\Omega_{03}$ [1,1,1,1]' ,':' ,2,'b'],] 
#     ylim = None
#     file_name = 'PcO2_CL.png'
#     plot_e3(results_dir,experiments,training_data_fractions,n_ds_max,xmax,c,nr_reps,file_name,ylim=ylim)
#     file_name = 'PcO2_CL_times.png'
#     plot_t3(results_dir,experiments,training_data_fractions,n_ds_max,xmax,c,nr_reps,file_name,ylim=ylim)


    
    ############################################
    
#     
#     #file_name = 'Plot_Pc.png'
#     #file_name = 'Plot_pc.png'
#     #file_name = 'Plot_TL_pc.png'
#     file_name = 'Plot_TL_Pn.png'
#     #file_name = 'Plot_ordinal.png'
#     plot_e2(results_dir,experiments,training_data_fractions,n_ds_max,xmax,c,nr_reps,file_name,ylim=ylim)
    
#     experiments = [['Pn'    ,None         ,None   , 1,'$Pn_{ds.ori}$'                             ,'-' ,2,'k'],]
#     file_name = 'Plot_Pn.png'
#     plot_e2(results_dir,experiments,training_data_fractions,n_ds_max,xmax,c,nr_reps,file_name,ylim=ylim)
#     
#     experiments = [['Pc',  'Pn',  'PT'    , [0,0,1,1,1,1,1,1],'$Pc_reusing_Pn[0,1,1]$'             ,'-' ,2,'k'],]
#     file_name = 'Plot_Pc_reusing_Pn_PT[011].png'
#     plot_e2(results_dir,experiments,training_data_fractions,n_ds_max,xmax,c,nr_reps,file_name,ylim=ylim)
#     
#     experiments = [['Pc',  'Pn',  'PT+FT' , [0,0,1,1,1,1,1,1],'$Pc_reusing_Pn[0,1,1]$'             ,'-' ,2,'k'],]
#     file_name = 'Plot_Pc_reusing_Pn_PT+FT[011].png'
#     plot_e2(results_dir,experiments,training_data_fractions,n_ds_max,xmax,c,nr_reps,file_name,ylim=ylim)
    
    
    
    
    
#     for experiment in experiments:
#         print 'experiment=',experiment
#         [approach, target_dataset, source_dataset, source_reuse_mode, retrain_ft_layers] = experiment
#         for training_data_fraction in training_data_fractions:
#             print 'training_data_fraction=',training_data_fraction
#             bl_or_tl(experiment,training_data_fraction,results_dir, params,nr_reps)
# 
#     # plotting
#     # experiments with k=1 
#      
#     nr_reps = 20#10#20
#     nr_processes = 5
#     #training_data_fractions = [0.10,0.20,0.30,0.40,0.50]
#     training_data_fractions = [0.30]
#     nr_rotations = 1
#      
#     n_ds_max = 3000 # 900#
#     c = 3
#     
#     # Tuning
#     experiments = [['Xs2_ori'    ,None         ,None   , 1,'$X2_{ds.ori}$'                             ,'-' ,2,'k'],]
#     ylim = None
#     file_name = 'plot_Xs2_non-canonical.png'
#     plot_e2(results_dir,experiments,training_data_fractions,n_ds_max,c,nr_reps,file_name,ylim=ylim)