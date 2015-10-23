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
    results_dir = 'results/test_MSTS[1111]/'
       
    nr_reps = 10  
    training_data_fractions = [1.00]
    
    
    xmax = 6000
    
    experiments = [['BL'   , 'mnist',                          None,     None    , None       ],   
                   ['MSTS' , 'mnist',     'chars74k_lowercase28x28',     'PT+FT' , [1,1,1,1]  ],
                   ['BL'   ,'chars74k_lowercase28x28',         None,     None    , None       ],    
                   ['MSTS' ,'chars74k_lowercase28x28',      'mnist',     'PT+FT' , [1,1,1,1]  ],]

    
    legend = ['BL',
              'STS:',
              'STS:',
              'STS:',
              'STS:', 
              'STS:',
              'STS:',
              'BL:',
              'STS:',
              'STS:',
              'STS:',
              'STS:',
              'STS:',
              'STS:',]
    
                  
    ylim = None
    file_name = 'STS_trail.png'
    plot_e2_sts(results_dir,experiments,legend,training_data_fractions,xmax,nr_reps,file_name,ylim=ylim)
    