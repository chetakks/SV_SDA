# NOTE: If openblas is installed (libopenblas-base and libopenblas-dev in
# Ubuntu) and multi-processing is needed (enabled by setting nr_processes>0),
# then the OMP_NUM_THREADS environment variable should be set to 1.
    
#from data_handling import *
#from execution import *
from plotting import *

import numpy
import string

if __name__ == "__main__":
    
    results_dir = '/home/aditya/store/Theano/STS_full/results/test_STS_full_drp/'
    
   
    nr_reps = 3#20 #5 #10
    training_data_fractions = [0.50,1.00]
    training_data_fractions = [0.05,0.10,0.15,0.20,0.30,0.40,0.50,1.00]
    
#     xmax = 19812 
#     experiments = [#['BL' , 'chars74k_lowercase28x28',        None,                              None    , None       ],
#                   ['BL' ,  'mnist',        None,                                                None    , None       ],
#                   #['TL' ,  'mnist',     'chars74k_lowercase28x28',                                  'R' , [1,1,0,1]  ],
#                   #['TL' ,  'mnist',     'chars74k_lowercase28x28',                                'R+D' , [1,1,0,1]  ],
#                   ['TL' ,  'mnist',     'chars74k_lowercase28x28',                                 'PT' , [1,1,0,1]  ],
#                   #['TL' ,  'mnist',     'chars74k_lowercase28x28',                               'PT+D' , [1,1,0,1]  ],
#                   ['TL' ,  'mnist',     'chars74k_lowercase28x28',                              'PT+FT' , [1,1,0,1]  ],
#                   #['TL' ,  'mnist',     'chars74k_lowercase28x28',                            'PT+FT+D' , [1,1,0,1]  ],
#                   ['STS', 'mnist',['TL','chars74k_lowercase28x28','mnist','PT+FT',[1,1,0,1] ],  'PT+FT' , [1,1,0,1]  ],]
#                      
#     legend = ['$\textbf{BL:}$ $P_L$',
#               #'$P_L,\Omega_{09}$ reusing $P_{LC},\Omega_{az}$,mode: $R$ [1101]',
#               #'$P_L,\Omega_{09}$ reusing $P_{LC},\Omega_{az}$,mode: $R+D$ [1101]',
#               '$\textbf{TLu:}$  $P_L$ reusing $P_{LC}$, $PT$',
#               #'$P_L,\Omega_{09}$ reusing $P_{LC},\Omega_{az}$,mode: $PT+D$ [1101]',
#               '$\textbf{TLs:}$ $P_L$ reusing $P_{LC}$, $PT+FT$',
#               #'$P_L,\Omega_{09}$ reusing $P_{LC},\Omega_{az}$,mode: $PT+FT+D$ [1101]', 
#               'STS: $P_L$ reusing $P_{LC}$ and reusing $P_L$, $PT+FT$' , ]
#        
#                      
#     ylim = None
#     file_name = 'STS_mnist_lc_3rep_ib.png' #'STS_mnist_lc.png'
#     plot_e2_sts(results_dir,experiments,legend,training_data_fractions,xmax,nr_reps,file_name,ylim=ylim)

#     xmax = 19812 
#     experiments = [#['BL' , 'chars74k_uppercase28x28',        None,                              None    , None       ],
#                   ['BL' ,  'mnist',        None,                                                None    , None       ],
#                   #['TL' ,  'mnist',     'chars74k_uppercase28x28',                                  'R' , [1,1,0,1]  ],
#                   #['TL' ,  'mnist',     'chars74k_uppercase28x28',                                'R+D' , [1,1,0,1]  ],
#                   ['TL' ,  'mnist',     'chars74k_uppercase28x28',                                 'PT' , [1,1,0,1]  ],
#                   #['TL' ,  'mnist',     'chars74k_uppercase28x28',                               'PT+D' , [1,1,0,1]  ],
#                   ['TL' ,  'mnist',     'chars74k_uppercase28x28',                              'PT+FT' , [1,1,0,1]  ],
#                   #['TL' ,  'mnist',     'chars74k_uppercase28x28',                            'PT+FT+D' , [1,1,0,1]  ],
#                   ['STS', 'mnist',['TL','chars74k_uppercase28x28','mnist','PT+FT',[1,1,0,1] ],  'PT+FT' , [1,1,0,1]  ],]
#                      
#    
#        
#     legend = ['$P_L$',
#               #'$P_L,\Omega_{09}$ reusing $P_{UC},\Omega_{az}$,mode: $R$ [1101]',
#               #'$P_L,\Omega_{09}$ reusing $P_{UC},\Omega_{az}$,mode: $R+D$ [1101]',
#               '$P_L$ reusing $P_{UC}$, $PT$',
#               #'$P_L,\Omega_{09}$ reusing $P_{UC},\Omega_{az}$,mode: $PT+D$ [1101]',
#               '$P_L$ reusing $P_{UC}$, $PT+FT$',
#               #'$P_L,\Omega_{09}$ reusing $P_{UC},\Omega_{az}$,mode: $PT+FT+D$ [1101]', 
#               'STS: $P_L$ reusing $P_{UC}$ and reusing $P_L$, $PT+FT$' , ]
#        
#                      
#     ylim = None
#     file_name = 'STS_mnist_uc_3rep_ib.png' #'STS_mnist_lc.png'
#     plot_e2_sts(results_dir,experiments,legend,training_data_fractions,xmax,nr_reps,file_name,ylim=ylim)

    # Plot relative errors for character recognition
    xmax = 19812 
    experiments =[['BL' ,  'chars74k_uppercase28x28',        None,                              None    , None       ],
                  ['TL' ,  'chars74k_uppercase28x28',     'mnist',                                 'PT' , [1,1,0,1]  ],
                  ['TL' ,  'chars74k_uppercase28x28',     'mnist',                              'PT+FT' , [1,1,0,1]  ],
                  ['STS',  'chars74k_uppercase28x28', ['TL','mnist', 'chars74k_uppercase28x28',  'PT+FT', [1,1,0,1]  ],
                                                                                                'PT+FT' , [1,1,0,1]  ],
                  ['BL' ,  'chars74k_lowercase28x28',        None,                              None    , None       ],
                  ['TL' ,  'chars74k_lowercase28x28',     'mnist',                                 'PT' , [1,1,0,1]  ],
                  ['TL' ,  'chars74k_lowercase28x28',     'mnist',                              'PT+FT' , [1,1,0,1]  ],
                  ['STS1',  'chars74k_lowercase28x28', ['TL','mnist', 'chars74k_lowercase28x28','PT+FT'  , [1,1,0,1] ],
                                                                                                'PT+FT' , [1,1,0,1]  ],
                  ]
      
    legend = [#'BL:              $P_{UC}$',
              'TLu: $P_L$  $\Rightarrow$ $P_{UC}$',
              'TLs: $P_L$  $\Rightarrow$ $P_{UC}$',
              'STS1: $P_{UC}$ $\Rightarrow$ $P_L$ $\Rightarrow$ $P_{UC}$' , 
              #'BL:              $P_{LC}$',
              'TLu: $P_L$  $\Rightarrow$ $P_{LC}$',
              'TLs: $P_L$  $\Rightarrow$ $P_{LC}$',
              'STS1: $P_{LC}$ $\Rightarrow$ $P_L$ $\Rightarrow$ $P_{LC}$',
              ]
         
                       
    ylim = None
    file_name = 'STS_uc+lc_mnist_3rep_relative.png'
    plot_r2_sts(results_dir,experiments,legend,training_data_fractions,xmax,nr_reps,file_name,ylim=ylim)
#     # Plot relative errors for character recognition with regions
#     xmax = 19812 
#     experiments =[['BL' ,  'chars74k_uppercase28x28',        None,                              None    , None       ],
#                   ['TLu' ,  'chars74k_uppercase28x28',     'mnist',                                 'PT' , [1,1,0,1]  ],
#                   ['TLs' ,  'chars74k_uppercase28x28',     'mnist',                              'PT+FT' , [1,1,0,1]  ],
#                   ['STS1',  'chars74k_uppercase28x28', ['TL','mnist', 'chars74k_uppercase28x28',  'PT+FT', [1,1,0,1]  ],
#                                                                                                 'PT+FT' , [1,1,0,1]  ],
#                   ['BL' ,  'chars74k_lowercase28x28',        None,                              None    , None       ],
#                   ['TLu' ,  'chars74k_lowercase28x28',     'mnist',                                 'PT' , [1,1,0,1]  ],
#                   ['TLs' ,  'chars74k_lowercase28x28',     'mnist',                              'PT+FT' , [1,1,0,1]  ],
#                   ['STS1',  'chars74k_lowercase28x28', ['TL','mnist', 'chars74k_lowercase28x28','PT+FT'  , [1,1,0,1] ],
#                                                                                                 'PT+FT' , [1,1,0,1]  ],
#                   ]
#      
#     legend = [#'BL:              $P_{UC}$',
#               'TLu: $P_L$  $\Rightarrow$ $P_{UC}$',
#               'TLs: $P_L$  $\Rightarrow$ $P_{UC}$',
#               'STS1: $P_{UC}$ $\Rightarrow$ $P_L$ $\Rightarrow$ $P_{UC}$' , 
#               #'BL:              $P_{LC}$',
#               'TLu: $P_L$  $\Rightarrow$ $P_{LC}$',
#               'TLs: $P_L$  $\Rightarrow$ $P_{LC}$',
#               'STS1: $P_{LC}$ $\Rightarrow$ $P_L$ $\Rightarrow$ $P_{LC}$',
#               ]
#         
#     ylim = None
#     file_name = 'STS_uc+lc_mnist_3rep_relative_avg.png'
#     plot_r3_sts(results_dir,experiments,legend,training_data_fractions,xmax,nr_reps,file_name,ylim=ylim)
    
#     # Plot relative errors for non-canonical object classification
#     xmax = 60000
#     experiments = [['BL' , 'sh2_1cs_2p_3o'  ,                   None,                              None    , None       ],
#                    ['BL' , 'sh2_1cs_2p_3o'  ,        'sh1_1cs_2p_3o',                              'Join'  , None       ],
#                    ['BL' , 'sh2_1cs_2p_3o'  ,      'sh202_1cs_2p_3o',                              'Join'  , None       ], 
#                     ['TL' ,   'sh2_1cs_2p_3o',        'sh1_1cs_2p_3o',                                 'PT' , [0,1,1,1]  ],
#                     ['TL' ,   'sh2_1cs_2p_3o',        'sh1_1cs_2p_3o',                              'PT+FT' , [0,1,1,1]  ],
#                     ['TL' ,   'sh2_1cs_2p_3o',      'sh202_1cs_2p_3o',                                 'PT' , [0,1,1,1]  ],
#                     ['TL' ,   'sh2_1cs_2p_3o',      'sh202_1cs_2p_3o',                              'PT+FT' , [0,1,1,1]  ],
#                     ['STS',   'sh2_1cs_2p_3o',  ['TL','sh1_1cs_2p_3o',     'sh2_1cs_2p_3o',         'PT+FT' , [0,1,1,1]  ],  
#                                                                                                     'PT+FT' , [0,1,1,1]  ],
#                     ['STS',   'sh2_1cs_2p_3o',  ['TL','sh202_1cs_2p_3o',   'sh2_1cs_2p_3o',         'PT+FT' , [0,1,1,1]  ],  
#                                                                                                     'PT+FT' , [0,1,1,1]  ],
#                    ]
#     
#     legend = [#'BL: $P_{sh2}$',
#               'cBL: $P_{sh2}$',
#               'cBL: $P_{sh2}$',
#             '$P_{sh2}$ by $P_{sh1}$, $PT$',
#             '$P_{sh2}$ by $P_{sh1}$, $PT+FT$',
#             '$P_{sh2}$ by $P_{sh2}$, $PT$',
#             '$P_{sh2}$ by $P_{sh2}$, $PT+FT$',
#             'STS: $P_{sh2}$ by $P_{sh1}$ and reusing $P_{sh2}$, $PT+FT$',
#             'STS: $P_{sh2}$ by $P_{sh2}$ and by $P_{sh2}$, $PT+FT$',
#               ]     
#     ylim = None
#     file_name = 'STS_sh2_sh1_and_sh202_relative.png'
#     plot_r2_sts(results_dir,experiments,legend,training_data_fractions,xmax,nr_reps,file_name,ylim=ylim)





#     xmax = 19812 
#     experiments =[['BL' ,  'chars74k_uppercase28x28',        None,                              None    , None       ],
#                   #['TL' ,  'chars74k_uppercase28x28',     'mnist',                                  'R' , [1,1,0,1]  ],
#                   #['TL' ,  'chars74k_uppercase28x28',     'mnist',                                'R+D' , [1,1,0,1]  ],
#                   ['TL' ,  'chars74k_uppercase28x28',     'mnist',                                 'PT' , [1,1,0,1]  ],
#                   #['TL' ,  'chars74k_uppercase28x28',     'mnist',                               'PT+D' , [1,1,0,1]  ],
#                   ['TL' ,  'chars74k_uppercase28x28',     'mnist',                              'PT+FT' , [1,1,0,1]  ],
#                   #['TL' ,  'chars74k_uppercase28x28',     'mnist',                            'PT+FT+D' , [1,1,0,1]  ],
#                   ['STS',  'chars74k_uppercase28x28', ['TL','mnist', 'chars74k_uppercase28x28',  'PT+FT', [1,1,0,1]  ],
#                                                                                                 'PT+FT' , [1,1,0,1]  ],
#                   #['STS',  'chars74k_uppercase28x28', ['TL','mnist', 'chars74k_uppercase28x28','PT+FT+D', [1,1,0,1]  ],
#                   #                                                                           'PT+FT+D' , [1,1,0,1]  ],
#                   ]
#                      
#        
# #     legend = ['$P_{UC},\Omega_{AZ}$',
# #               #'$P_{UC},\Omega_{AZ}$ reusing $P_L,\Omega_{09}$,mode: $R$ [1101]',
# #               #'$P_{UC},\Omega_{AZ}$ reusing $P_L,\Omega_{09}$,mode: $R+D$ [1101]',
# #               '$P_{UC},\Omega_{AZ}$ reusing $P_L,\Omega_{09}$,mode: $PT$ [1101]',
# #               #'$P_{UC},\Omega_{AZ}$ reusing $P_L,\Omega_{09}$,mode: $PT+D$ [1101]',
# #               '$P_{UC},\Omega_{AZ}$ reusing $P_L,\Omega_{09}$,mode: $PT+FT$ [1101]',
# #               #'$P_{UC},\Omega_{AZ}$ reusing $P_L,\Omega_{09}$,mode: $PT+FT+D$ [1101]', 
# #               'STS: $P_{UC}$ reusing $P_L$ and reusing $P_{UC}$,mode: $PT+FT$ [1101]' , 
# #               #'STS: $P_{UC}$ reusing $P_L$ and reusing $P_{UC}$,mode: $PT+FT+D$ [1101]' ,
# #               ]
#       
#     legend = ['BL:              $P_{UC}$',
#               #'$P_{UC},\Omega_{AZ}$ reusing $P_L,\Omega_{09}$,mode: $R$ [1101]',
#               #'$P_{UC},\Omega_{AZ}$ reusing $P_L,\Omega_{09}$,mode: $R+D$ [1101]',
#               'TLu: $P_L$  $\Rightarrow$ $P_{UC}$',
#               #'$P_{UC},\Omega_{AZ}$ reusing $P_L,\Omega_{09}$,mode: $PT+D$ [1101]',
#               'TLs: $P_L$  $\Rightarrow$ $P_{UC}$',
#               #'$P_{UC},\Omega_{AZ}$ reusing $P_L,\Omega_{09}$,mode: $PT+FT+D$ [1101]', 
#               'STS: $P_{UC}$ $\Rightarrow$ $P_L$ $\Rightarrow$ $P_{UC}$' , 
#               #'STS: $P_{UC}$ reusing $P_L$ and reusing $P_{UC}$,mode: $PT+FT+D$ [1101]' ,
#               ]
#        
#                      
#     ylim = None
#     file_name = 'STS_uc_mnist_3rep_ibp2.png'
#     plot_e2_sts(results_dir,experiments,legend,training_data_fractions,xmax,nr_reps,file_name,ylim=ylim)
# #     ##################      ##################      ##################      ##################      ##################    
# #     
# #      
#     xmax = 19812
#     experiments =[['BL' ,  'chars74k_lowercase28x28',        None,                              None    , None       ],
#                   #['TL' ,  'chars74k_lowercase28x28',     'mnist',                                  'R' , [1,1,0,1]  ],
#                   #['TL' ,  'chars74k_lowercase28x28',     'mnist',                                'R+D' , [1,1,0,1]  ],
#                   ['TL' ,  'chars74k_lowercase28x28',     'mnist',                                 'PT' , [1,1,0,1]  ],
#                   #['TL' ,  'chars74k_lowercase28x28',     'mnist',                               'PT+D' , [1,1,0,1]  ],
#                   ['TL' ,  'chars74k_lowercase28x28',     'mnist',                              'PT+FT' , [1,1,0,1]  ],
#                   #['TL' ,  'chars74k_lowercase28x28',     'mnist',                            'PT+FT+D' , [1,1,0,1]  ],
#                   ['STS1',  'chars74k_lowercase28x28', ['TL','mnist', 'chars74k_lowercase28x28','PT+FT'  , [1,1,0,1]  ],
#                                                                                                 'PT+FT' , [1,1,0,1]  ],
#                   #['STS',  'chars74k_lowercase28x28', ['TL','mnist', 'chars74k_lowercase28x28','PT+FT+D', [1,1,0,1]  ],
#                    #                                                                            'PT+FT+D', [1,1,0,1]  ],
#                    ]
#                       
#         
# #     legend = ['$P_{LC},\Omega_{az}$',
# #               #'$P_{LC},\Omega_{az}$ reusing $P_L,\Omega_{09}$,mode: $R$ [1101]',
# #               #'$P_{LC},\Omega_{az}$ reusing $P_L,\Omega_{09}$,mode: $R+D$ [1101]',
# #               '$P_{LC},\Omega_{az}$ reusing $P_L,\Omega_{09}$,mode: $PT$ [1101]',
# #               #'$P_{LC},\Omega_{az}$ reusing $P_L,\Omega_{09}$,mode: $PT+D$ [1101]',
# #               '$P_{LC},\Omega_{az}$ reusing $P_L,\Omega_{09}$,mode: $PT+FT$ [1101]',
# #               #'$P_{LC},\Omega_{az}$ reusing $P_L,\Omega_{09}$,mode: $PT+FT+D$ [1101]', 
# #               'STS: $P_{LC}$ reusing $P_L$ and reusing $P_{LC}$,mode: $PT+FT$ [1101]' ,
# #               #'STS: $P_{LC}$ reusing $P_L$ and reusing $P_{LC}$,mode: $PT+FT+D$ [1101]' , 
# #               ]
#   
#     legend = ['BL:              $P_{LC}$',
#               #'$P_{LC},\Omega_{az}$ reusing $P_L,\Omega_{09}$,mode: $R$ [1101]',
#               #'$P_{LC},\Omega_{az}$ reusing $P_L,\Omega_{09}$,mode: $R+D$ [1101]',
#               'TLu: $P_L$  $\Rightarrow$ $P_{LC}$',
#               #'$P_{LC},\Omega_{az}$ reusing $P_L,\Omega_{09}$,mode: $PT+D$ [1101]',
#               'TLs: $P_L$  $\Rightarrow$ $P_{LC}$',
#               #'$P_{LC},\Omega_{az}$ reusing $P_L,\Omega_{09}$,mode: $PT+FT+D$ [1101]', 
#               'STS1: $P_{LC}$ $\Rightarrow$ $P_L$ $\Rightarrow$ $P_{LC}$',
#               #'STS: $P_{LC}$ reusing $P_L$ and reusing $P_{LC}$,mode: $PT+FT+D$ [1101]' , 
#               ]
#                       
#     ylim = None
#     file_name = 'STS_lc_mnist_3rep_ibp2.png'
#     plot_e2_sts(results_dir,experiments,legend,training_data_fractions,xmax,nr_reps,file_name,ylim=ylim)
    
    
#     xmax = 60000
#     experiments = [['BL' , 'sh2_1cs_2p_3o'  ,                   None,                              None    , None       ],                  
#                    ['TL' ,   'sh2_1cs_2p_3o',        'sh1_1cs_2p_3o',                                 'PT' , [0,1,1,1]  ],
#                    ['TL' ,   'sh2_1cs_2p_3o',        'sh1_1cs_2p_3o',                              'PT+FT' , [0,1,1,1]  ],
#                    ['TL' ,   'sh2_1cs_2p_3o',      'sh202_1cs_2p_3o',                                 'PT' , [0,1,1,1]  ],
#                    ['TL' ,   'sh2_1cs_2p_3o',      'sh202_1cs_2p_3o',                              'PT+FT' , [0,1,1,1]  ],
#                    ['STS',   'sh2_1cs_2p_3o',  ['TL','sh1_1cs_2p_3o',     'sh2_1cs_2p_3o',         'PT+FT' , [0,1,1,1]  ],  
#                                                                                                    'PT+FT' , [0,1,1,1]  ],
#                    ['STS',   'sh2_1cs_2p_3o',  ['TL','sh202_1cs_2p_3o',   'sh2_1cs_2p_3o',         'PT+FT' , [0,1,1,1]  ],  
#                                                                                                    'PT+FT' , [0,1,1,1]  ],
#                    ]
#     legend = ['$P_{sh2}$',
#               '$P_{sh2}$ by $P_{sh1}$, $PT$',
#               '$P_{sh2}$ by $P_{sh1}$, $PT+FT$',
#               '$P_{sh2}$ by $P_{sh2}$, $PT$',
#               '$P_{sh2}$ by $P_{sh2}$, $PT+FT$',
#               'STS: $P_{sh2}$ by $P_{sh1}$ and reusing $P_{sh2}$, $PT+FT$',
#               'STS: $P_{sh2}$ by $P_{sh2}$ and by $P_{sh2}$, $PT+FT$',
#               ]     
#     ylim = None
#     file_name = 'STS_sh2_sh1_and_sh202.png'
#     plot_e2_sts(results_dir,experiments,legend,training_data_fractions,xmax,nr_reps,file_name,ylim=ylim)
    
#     xmax = 60000
#     experiments = [['BL' , 'sh2_1cs_2p_3o'  ,                   None,                              None    , None       ],                  
#                    ['BL' , 'sh2_1cs_2p_3o'  ,        'sh1_1cs_2p_3o',                              'Join'  , None       ],
#                    ['BL' , 'sh2_1cs_2p_3o'  ,      'sh202_1cs_2p_3o',                              'Join'  , None       ], 
#                    ['TL' ,   'sh2_1cs_2p_3o',        'sh1_1cs_2p_3o',                                  'R' , [0,1,1,1]  ],
#                    ['TL' ,   'sh2_1cs_2p_3o',        'sh1_1cs_2p_3o',                                'R+D' , [0,1,1,1]  ],
#                    ['TL' ,   'sh2_1cs_2p_3o',        'sh1_1cs_2p_3o',                                 'PT' , [0,1,1,1]  ],
#                    ['TL' ,   'sh2_1cs_2p_3o',        'sh1_1cs_2p_3o',                               'PT+D' , [0,1,1,1]  ],
#                    ['TL' ,   'sh2_1cs_2p_3o',        'sh1_1cs_2p_3o',                              'PT+FT' , [0,1,1,1]  ],
#                    ['TL' ,   'sh2_1cs_2p_3o',        'sh1_1cs_2p_3o',                            'PT+FT+D' , [0,1,1,1]  ],
#                    ['TL' ,   'sh2_1cs_2p_3o',      'sh202_1cs_2p_3o',                                  'R' , [0,1,1,1]  ],
#                    ['TL' ,   'sh2_1cs_2p_3o',      'sh202_1cs_2p_3o',                                'R+D' , [0,1,1,1]  ],
#                    ['TL' ,   'sh2_1cs_2p_3o',      'sh202_1cs_2p_3o',                                 'PT' , [0,1,1,1]  ],
#                    ['TL' ,   'sh2_1cs_2p_3o',      'sh202_1cs_2p_3o',                               'PT+D' , [0,1,1,1]  ],
#                    ['TL' ,   'sh2_1cs_2p_3o',      'sh202_1cs_2p_3o',                              'PT+FT' , [0,1,1,1]  ],
#                    ['TL' ,   'sh2_1cs_2p_3o',      'sh202_1cs_2p_3o',                            'PT+FT+D' , [0,1,1,1]  ],
#                    ['STS',   'sh2_1cs_2p_3o',  ['TL','sh1_1cs_2p_3o',     'sh2_1cs_2p_3o',         'PT+FT' , [0,1,1,1]  ],  
#                                                                                                    'PT+FT' , [0,1,1,1]  ],
#                    ['STS',   'sh2_1cs_2p_3o',  ['TL','sh1_1cs_2p_3o',     'sh2_1cs_2p_3o',       'PT+FT+D' , [0,1,1,1]  ],  
#                                                                                                  'PT+FT+D' , [0,1,1,1]  ],
#                    ['STS',   'sh2_1cs_2p_3o',  ['TL','sh202_1cs_2p_3o',   'sh2_1cs_2p_3o',         'PT+FT' , [0,1,1,1]  ],  
#                                                                                                    'PT+FT' , [0,1,1,1]  ],
#                    ['STS',   'sh2_1cs_2p_3o',  ['TL','sh202_1cs_2p_3o',   'sh2_1cs_2p_3o',       'PT+FT+D' , [0,1,1,1]  ],  
#                                                                                                  'PT+FT+D' , [0,1,1,1]  ],]
#  
#      
#     legend = ['$P_{sh2},\Omega_{02}$',
#               '$P_{sh2},\Omega_{02}$ join $P_{sh1},\Omega_{02}$',
#               '$P_{sh2},\Omega_{02}$ join $P_{sh2},\Omega_{01}$',
#               '$P_{sh2},\Omega_{02}$ reusing $P_{sh1},\Omega_{02}$,mode: $R$ [0111]',
#               '$P_{sh2},\Omega_{02}$ reusing $P_{sh1},\Omega_{02}$,mode: $R+D$ [0111]',
#               '$P_{sh2},\Omega_{02}$ reusing $P_{sh1},\Omega_{02}$,mode: $PT$ [0111]',
#               '$P_{sh2},\Omega_{02}$ reusing $P_{sh1},\Omega_{02}$,mode: $PT+D$ [0111]',
#               '$P_{sh2},\Omega_{02}$ reusing $P_{sh1},\Omega_{02}$,mode: $PT+FT$ [0111]',
#               '$P_{sh2},\Omega_{02}$ reusing $P_{sh1},\Omega_{02}$,mode: $PT+FT+D$ [0111]', 
#               '$P_{sh2},\Omega_{02}$ reusing $P_{sh2},\Omega_{01}$,mode: $R$ [0111]',
#               '$P_{sh2},\Omega_{02}$ reusing $P_{sh2},\Omega_{01}$,mode: $R+D$ [0111]',
#               '$P_{sh2},\Omega_{02}$ reusing $P_{sh2},\Omega_{01}$,mode: $PT$ [0111]',
#               '$P_{sh2},\Omega_{02}$ reusing $P_{sh2},\Omega_{01}$,mode: $PT+D$ [0111]',
#               '$P_{sh2},\Omega_{02}$ reusing $P_{sh2},\Omega_{01}$,mode: $PT+FT$ [0111]',
#               '$P_{sh2},\Omega_{02}$ reusing $P_{sh2},\Omega_{01}$,mode: $PT+FT+D$ [0111]',
#               'STS: $P_{sh2},\Omega_{02}$ reusing $P_{sh1},\Omega_{02}$ and reusing $P_{sh2},\Omega_{02}$,mode: $PT+FT$ [0111]',
#               'STS: $P_{sh2},\Omega_{02}$ reusing $P_{sh1},\Omega_{02}$ and reusing $P_{sh2},\Omega_{02}$,mode: $PT+FT+D$ [0111]',
#               'STS: $P_{sh2},\Omega_{02}$ reusing $P_{sh2},\Omega_{01}$ and reusing $P_{sh2},\Omega_{02}$,mode: $PT+FT$ [0111]',
#               'STS: $P_{sh2},\Omega_{02}$ reusing $P_{sh2},\Omega_{01}$ and reusing $P_{sh2},\Omega_{02}$,mode: $PT+FT+D$ [0111]', ]
#      
#                    
#     ylim = None
#     file_name = 'STS_sh2_sh1_and_sh202.png'
#     plot_e2_sts(results_dir,experiments,legend,training_data_fractions,xmax,nr_reps,file_name,ylim=ylim)