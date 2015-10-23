# NOTE: If openblas is installed (libopenblas-base and libopenblas-dev in
# Ubuntu) and multi-processing is needed (enabled by setting nr_processes>0),
# then the OMP_NUM_THREADS environment variable should be set to 1.
    
#from data_handling import *
#from execution import *
from plotting import *

import numpy
import string

if __name__ == "__main__":
    
    results_dir = 'results/test_STS_full_drp/'
    
   
    nr_reps = 3#20 #5 #10
    #repeat = nr_reps
    
    #training_data_fractions = [1.00,0.50,0.40,0.30,0.20,0.15,0.10,0.05]
    training_data_fractions = [0.50,1.00]
    training_data_fractions = [0.05,0.10,0.15,0.20,0.30,0.40,0.50,1.00]
    #n_ds_max = 5000
    xmax = 6000  # 900#
    #c = 3  #5

    xmax = 60000
    
    experiments = [#['BL' , 'chars74k_uppercase28x28',        None,                              None    , None       ],
                  ['BL' ,  'mnist',        None,                                                None    , None       ],
                  ['TL' ,  'mnist',     'chars74k_uppercase28x28',                                  'R' , [1,1,0,1]  ],
                  ['TL' ,  'mnist',     'chars74k_uppercase28x28',                                'R+D' , [1,1,0,1]  ],
                  ['TL' ,  'mnist',     'chars74k_uppercase28x28',                                 'PT' , [1,1,0,1]  ],
                  ['TL' ,  'mnist',     'chars74k_uppercase28x28',                               'PT+D' , [1,1,0,1]  ],
                  ['TL' ,  'mnist',     'chars74k_uppercase28x28',                              'PT+FT' , [1,1,0,1]  ],
                  ['TL' ,  'mnist',     'chars74k_uppercase28x28',                            'PT+FT+D' , [1,1,0,1]  ],
                  ['STS', 'mnist',['TL','chars74k_uppercase28x28','mnist','PT+FT',[1,1,0,1] ],  'PT+FT' , [1,1,0,1]  ],]
                  

    
    legend = ['$P_L,\Omega_{09}$',
              '$P_L,\Omega_{09}$ reusing $P_{UC},\Omega_{AZ}$,mode: $R$ [1101]',
              '$P_L,\Omega_{09}$ reusing $P_{UC},\Omega_{AZ}$,mode: $R+D$ [1101]',
              '$P_L,\Omega_{09}$ reusing $P_{UC},\Omega_{AZ}$,mode: $PT$ [1101]',
              '$P_L,\Omega_{09}$ reusing $P_{UC},\Omega_{AZ}$,mode: $PT+D$ [1101]',
              '$P_L,\Omega_{09}$ reusing $P_{UC},\Omega_{AZ}$,mode: $PT+FT$ [1101]',
              '$P_L,\Omega_{09}$ reusing $P_{UC},\Omega_{AZ}$,mode: $PT+FT+D$ [1101]', 
              'STS: $P_L$ reusing $P_{UC}$ and reusing $P_L$,mode: $PT+FT$ [1101]' , ]
    
                  
    ylim = None
    file_name = 'STS_mnist_uc.png'
    plot_e2_sts(results_dir,experiments,legend,training_data_fractions,xmax,nr_reps,file_name,ylim=ylim)
    
    ##################      ##################      ##################      ##################      
    
    experiments = [#['BL' , 'chars74k_lowercase28x28',        None,                              None    , None       ],
                  ['BL' ,  'mnist',        None,                                                None    , None       ],
                  ['TL' ,  'mnist',     'chars74k_lowercase28x28',                                  'R' , [1,1,0,1]  ],
                  ['TL' ,  'mnist',     'chars74k_lowercase28x28',                                'R+D' , [1,1,0,1]  ],
                  ['TL' ,  'mnist',     'chars74k_lowercase28x28',                                 'PT' , [1,1,0,1]  ],
                  ['TL' ,  'mnist',     'chars74k_lowercase28x28',                               'PT+D' , [1,1,0,1]  ],
                  ['TL' ,  'mnist',     'chars74k_lowercase28x28',                              'PT+FT' , [1,1,0,1]  ],
                  ['TL' ,  'mnist',     'chars74k_lowercase28x28',                            'PT+FT+D' , [1,1,0,1]  ],
                  ['STS', 'mnist',['TL','chars74k_lowercase28x28','mnist','PT+FT',[1,1,0,1] ],  'PT+FT' , [1,1,0,1]  ],]
                  

    
    legend = ['$P_L,\Omega_{09}$',
              '$P_L,\Omega_{09}$ reusing $P_{LC},\Omega_{az}$,mode: $R$ [1101]',
              '$P_L,\Omega_{09}$ reusing $P_{LC},\Omega_{az}$,mode: $R+D$ [1101]',
              '$P_L,\Omega_{09}$ reusing $P_{LC},\Omega_{az}$,mode: $PT$ [1101]',
              '$P_L,\Omega_{09}$ reusing $P_{LC},\Omega_{az}$,mode: $PT+D$ [1101]',
              '$P_L,\Omega_{09}$ reusing $P_{LC},\Omega_{az}$,mode: $PT+FT$ [1101]',
              '$P_L,\Omega_{09}$ reusing $P_{LC},\Omega_{az}$,mode: $PT+FT+D$ [1101]', 
              'STS: $P_L$ reusing $P_{LC}$ and reusing $P_L$,mode: $PT+FT+D$ [1101]' , ]
    
                  
    ylim = None
    file_name = 'STS_mnist_lc.png'
    plot_e2_sts(results_dir,experiments,legend,training_data_fractions,xmax,nr_reps,file_name,ylim=ylim)
    
    ##################      ##################      ##################      ##################     
    
    
    
    xmax = 19812 
    
    experiments =[['BL' ,  'chars74k_uppercase28x28',        None,                              None    , None       ],
                  ['TL' ,  'chars74k_uppercase28x28',     'mnist',                                  'R' , [1,1,0,1]  ],
                  ['TL' ,  'chars74k_uppercase28x28',     'mnist',                                'R+D' , [1,1,0,1]  ],
                  ['TL' ,  'chars74k_uppercase28x28',     'mnist',                                 'PT' , [1,1,0,1]  ],
                  ['TL' ,  'chars74k_uppercase28x28',     'mnist',                               'PT+D' , [1,1,0,1]  ],
                  ['TL' ,  'chars74k_uppercase28x28',     'mnist',                              'PT+FT' , [1,1,0,1]  ],
                  ['TL' ,  'chars74k_uppercase28x28',     'mnist',                            'PT+FT+D' , [1,1,0,1]  ],
                  ['STS',  'chars74k_uppercase28x28', ['TL','mnist', 'chars74k_uppercase28x28',  'PT+FT', [1,1,0,1]  ],
                                                                                                'PT+FT' , [1,1,0,1]  ],
                  ['STS',  'chars74k_uppercase28x28', ['TL','mnist', 'chars74k_uppercase28x28','PT+FT+D', [1,1,0,1]  ],
                                                                                              'PT+FT+D' , [1,1,0,1]  ],]
                  
    
    legend = ['$P_{UC},\Omega_{AZ}$',
              '$P_{UC},\Omega_{AZ}$ reusing $P_L,\Omega_{09}$,mode: $R$ [1101]',
              '$P_{UC},\Omega_{AZ}$ reusing $P_L,\Omega_{09}$,mode: $R+D$ [1101]',
              '$P_{UC},\Omega_{AZ}$ reusing $P_L,\Omega_{09}$,mode: $PT$ [1101]',
              '$P_{UC},\Omega_{AZ}$ reusing $P_L,\Omega_{09}$,mode: $PT+D$ [1101]',
              '$P_{UC},\Omega_{AZ}$ reusing $P_L,\Omega_{09}$,mode: $PT+FT$ [1101]',
              '$P_{UC},\Omega_{AZ}$ reusing $P_L,\Omega_{09}$,mode: $PT+FT+D$ [1101]', 
              'STS: $P_{UC}$ reusing $P_L$ and reusing $P_{UC}$,mode: $PT+FT$ [1101]' , 
              'STS: $P_{UC}$ reusing $P_L$ and reusing $P_{UC}$,mode: $PT+FT+D$ [1101]' ,]
    
                  
    ylim = None
    file_name = 'STS_uc_mnist.png'
    plot_e2_sts(results_dir,experiments,legend,training_data_fractions,xmax,nr_reps,file_name,ylim=ylim)
    
    
    ##################      ##################      ##################      ##################      ##################    
    
    
      
    
    
     
     
    experiments =[['BL' ,  'chars74k_lowercase28x28',        None,                              None    , None       ],
                  ['TL' ,  'chars74k_lowercase28x28',     'mnist',                                  'R' , [1,1,0,1]  ],
                  ['TL' ,  'chars74k_lowercase28x28',     'mnist',                                'R+D' , [1,1,0,1]  ],
                  ['TL' ,  'chars74k_lowercase28x28',     'mnist',                                 'PT' , [1,1,0,1]  ],
                  ['TL' ,  'chars74k_lowercase28x28',     'mnist',                               'PT+D' , [1,1,0,1]  ],
                  ['TL' ,  'chars74k_lowercase28x28',     'mnist',                              'PT+FT' , [1,1,0,1]  ],
                  ['TL' ,  'chars74k_lowercase28x28',     'mnist',                            'PT+FT+D' , [1,1,0,1]  ],
                  ['STS',  'chars74k_lowercase28x28', ['TL','mnist', 'chars74k_lowercase28x28','PT+FT'  , [1,1,0,1]  ],
                                                                                                'PT+FT' , [1,1,0,1]  ],
                  ['STS',  'chars74k_lowercase28x28', ['TL','mnist', 'chars74k_lowercase28x28','PT+FT+D', [1,1,0,1]  ],
                                                                                               'PT+FT+D', [1,1,0,1]  ],]
                  
    
    legend = ['$P_{LC},\Omega_{az}$',
              '$P_{LC},\Omega_{az}$ reusing $P_L,\Omega_{09}$,mode: $R$ [1101]',
              '$P_{LC},\Omega_{az}$ reusing $P_L,\Omega_{09}$,mode: $R+D$ [1101]',
              '$P_{LC},\Omega_{az}$ reusing $P_L,\Omega_{09}$,mode: $PT$ [1101]',
              '$P_{LC},\Omega_{az}$ reusing $P_L,\Omega_{09}$,mode: $PT+D$ [1101]',
              '$P_{LC},\Omega_{az}$ reusing $P_L,\Omega_{09}$,mode: $PT+FT$ [1101]',
              '$P_{LC},\Omega_{az}$ reusing $P_L,\Omega_{09}$,mode: $PT+FT+D$ [1101]', 
              'STS: $P_{LC}$ reusing $P_L$ and reusing $P_{LC}$,mode: $PT+FT$ [1101]' ,
              'STS: $P_{LC}$ reusing $P_L$ and reusing $P_{LC}$,mode: $PT+FT+D$ [1101]' , ]
    
                  
    ylim = None
    file_name = 'STS_lc_mnist.png'
    plot_e2_sts(results_dir,experiments,legend,training_data_fractions,xmax,nr_reps,file_name,ylim=ylim)