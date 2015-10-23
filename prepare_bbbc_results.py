import pickle
import numpy
from scipy import stats
from sklearn.metrics import confusion_matrix

def bbbc_collective_results(target_dataset, y_test_pred, y_test ):
    PC = 'hpc'
    if PC == 'pc':
        results_dir = 'results/BL_bbbc_feat4/'
        if target_dataset == 'bbbc+moa':
            f1 = open(results_dir+'meta_data/bbbc+moa_cell_metadata.pkl', 'rb')
            f2 = open(results_dir+'meta_data/bbbc+moa_tes_met.pkl', 'rb')
        elif target_dataset == 'bbbc+comp':
            f1 = open(results_dir+'meta_data/bbbc+comp_cell_metadata.pkl', 'rb')
            f2 = open(results_dir+'meta_data/bbbc+comp_tes_met.pkl', 'rb')
    elif PC == 'hpc':
        results_dir = '/home/aditya/store/Datasets/pickled/bbbc+feat4/'
        if target_dataset == 'bbbc+moa':
            f1 = open(results_dir+'bbbc+moa_cell_metadata.pkl', 'rb')
            f2 = open(results_dir+'bbbc+moa_tes_met.pkl', 'rb')
        elif target_dataset == 'bbbc+comp':
            f1 = open(results_dir+'bbbc+comp_cell_metadata.pkl', 'rb')
            f2 = open(results_dir+'bbbc+comp_tes_met.pkl', 'rb')
        
    
    
    cell_metadata = pickle.load(f1)
    cell_metadata = cell_metadata.astype(int)
    f1.close()
    tes_met = pickle.load(f2)
    tes_met = tes_met.astype(int)
    f2.close()
    
    y_test_pred = y_test_pred.flatten()
    
    #print 'len y_test_pred', len(y_test_pred)# len(y_test_pred)
    #print 'type y_test_pred', type(y_test_pred) #type(y_test_pred)
    
    print 'processing the individual cells ... '
    #identify the full image and cell labels
    
    # match the batch processing prediction to meta data
    tes_met = tes_met[0:len(y_test_pred)]
    
    Image_nrs = set(tes_met[0:,1])
    Total_tables = 0
    for Image_nr in Image_nrs:
        #print 'Image_nr', Image_nr
        Tabel_nrs = set(tes_met[numpy.where(tes_met[0:,1] == Image_nr)][0:,0])
        Total_tables = Total_tables + len(Tabel_nrs)
    
    predictions = numpy.zeros(([Total_tables,4]), dtype=numpy.int)   
    
    count = 0
    for Image_nr in Image_nrs:
        #print 'Image_nr', Image_nr
        im = tes_met[numpy.where(tes_met[0:,1] == Image_nr)]
        #print im
        
        Tabel_nrs = set(im[0:,0])
        for Tabel_nr in Tabel_nrs:
            #print 'Tabel_nr', Tabel_nr
            #print 'Image_nr', Image_nr
            cells = tes_met[(numpy.where((tes_met[0:,0] == Tabel_nr) & (tes_met[0:,1] == Image_nr)))] #[0,2]
            #print 'cells', set(cells[0:,2]) #cells
            cell_nrs = set(cells[0:,2]) 
            
            cell_value = numpy.zeros(([len(cell_nrs),2]), dtype=numpy.int)  
            #cell_value = numpy.zeros([len(cell_nrs),1])
            for idx, cell_nr in enumerate(cell_nrs):
                #print 'cell', cell_nr
                #print idx 
                #print
                
                if target_dataset == 'bbbc+moa':
                    cell_value[idx, 0] = tes_met[(numpy.where((tes_met[0:,0] == Tabel_nr) & (tes_met[0:,1] == Image_nr) & (tes_met[0:,2] == cell_nr)))][0,3]
                    cell_value[idx, 1] = y_test_pred[(numpy.where((tes_met[0:,0] == Tabel_nr) & (tes_met[0:,1] == Image_nr) & (tes_met[0:,2] == cell_nr)))][0] #,0]
                elif target_dataset == 'bbbc+comp':
                    cell_value[idx, 0] = tes_met[(numpy.where((tes_met[0:,0] == Tabel_nr) & (tes_met[0:,1] == Image_nr) & (tes_met[0:,2] == cell_nr)))][0,4]
                    cell_value[idx, 1] = y_test_pred[(numpy.where((tes_met[0:,0] == Tabel_nr) & (tes_met[0:,1] == Image_nr) & (tes_met[0:,2] == cell_nr)))][0] #,0]
                #print y_test_pred[(numpy.where((tes_met[0:,0] == Tabel_nr) & (tes_met[0:,1] == Image_nr) & (tes_met[0:,2] == cell_nr)))][0,0]
                #cell_value[idx] = y_test_pred[(numpy.where((tes_met[0:,0] == Tabel_nr) & (tes_met[0:,1] == Image_nr) & (tes_met[0:,2] == cell_nr)))][0,0]
    #        
            cell_value_true = cell_value[0:,0]
            cell_value_pred = cell_value[0:,1]
    
    #         print 'Prediction of cell in a Image', Image_nr, Tabel_nrs
            #print 'Prediction of cells in a Image', stats.itemfreq(cell_value_pred) 
    #         print 'Highest prediction in a Image', stats.itemfreq(cell_value_pred)[0,0] 
            freq = stats.itemfreq(cell_value_pred)
            #freq[numpy.argmax(freq[0:,1])][0].astype(int)
            
            #print Predictions
            #print Tabel_nr, Image_nr, stats.itemfreq(cell_value_true)[0,0].astype(int), freq[numpy.argmax(freq[0:,1])][0].astype(int)
            predictions[count] = Tabel_nr, Image_nr, stats.itemfreq(cell_value_true)[0,0].astype(int), freq[numpy.argmax(freq[0:,1])][0].astype(int)
    
            
            count += 1
#             print 'count', count
#             if count > 1:
#                 break  
            
    
    true_val = predictions[0:,2]
    pred_val = predictions[0:,3]
    from sklearn.metrics import accuracy_score
    avg_acc = accuracy_score(true_val, pred_val) 
    avg_cm = confusion_matrix(true_val, pred_val)
#     print 'Average the prediction of individual cells'
#     print 'Accuracy', avg_acc
#     print 'Test error', 1-avg_acc   
#     print
#     print
#     print 'Prediction of individual cells'
    aa = y_test_pred.flatten()
    bb = y_test.flatten()
    ind_acc = accuracy_score(bb, aa)
    ind_cm = confusion_matrix(bb,aa)
    #print 'Accuracy', ind_acc
    #print 'Test error', 1-ind_acc   
    
    
    return avg_acc, avg_cm, ind_acc, ind_cm