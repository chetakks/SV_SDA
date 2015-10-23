from plotting import *

import numpy
import string

if __name__ == "__main__":
    
    results_dir = 'results/test_BBBC_MSTS[1111]moa+com/'
    results_dir = 'results/test_BBBC_MSTS[0011]moa+com/'

    
    nr_reps = 20
    training_data_fractions = [1.00]
    
    experiments = [#['BL'   ,'bbbc+comp',     None,        None    , None       ],           
               #['MSTS' ,'bbbc+moa',  'bbbc+comp',     'PT+FT' , [1,1,1,1]  ],
               #['MSTS' ,'bbbc+comp',  'bbbc+moa',     'PT+FT' , [1,1,1,1]  ],
               ['MSTS' ,'bbbc+comp',  'bbbc+moa',     'PT+FT' , [0,0,1,1]  ],
               ['MSTS' ,'bbbc+moa',  'bbbc+comp',     'PT+FT' , [0,0,1,1]  ],
               ]
    
    # Avg errors for BBBC                   
    ylim = None
    file_name = 'MSTS_20rep.png'
    plot_BBBC_sts2(results_dir,experiments,training_data_fractions,nr_reps,file_name,ylim=ylim)

    