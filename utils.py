#Imports
import numpy as np

fe_cqt_fd 			 = 'Fe/cqt' #2
fe_logmel_kong_fd 	 = 'Fe/logmel_kong'#4
fe_logmel_libd_fd 	 = 'Fe/logmel_lib_delta'#7
fe_mel_fd 			 = 'Fe/mel'#9

fe_eva_cqt_fd 			 = 'Fe_eva/cqt' #2
fe_eva_logmel_kong_fd 	 = 'Fe_eva/logmel_kong'#4
fe_eva_logmel_libd_fd 	 = 'Fe_eva/logmel_lib_delta'#7
fe_eva_mel_fd 			 = 'Fe_eva/mel'#9


meta_train_csv      = 'meta_csvs/development_chunks_o_refined.csv'
meta_test_csv       = 'meta_csvs/evaluation_chunks_o_refined.csv' #eva_csv_path
label_csv           = 'label_csvs'


# 1 of 7 acoustic label
labels = [ 'c', 'm', 'f', 'v', 'p', 'b', 'S' ]

           
lb_to_id = { lb:id for id, lb in enumerate(labels) }
id_to_lb = { id:lb for id, lb in enumerate(labels) }

fs = 16000.     # sample rate
win = 1024.     # fft window size
n_folds = 5

feature=['logmel_kong','mel','cqt']
model='ensemble'
agg_num=15           # Agg Number(Integer) Number of frames
hop=10
n_labels = len( labels )

def feature1():
    fe_fd = fe_cqt_fd   # cqt
    fe_fd_e=fe_eva_cqt_fd
    text="cqt"
    return fe_fd,fe_fd_e,text

def feature2():
    fe_fd = fe_logmel_kong_fd   # logmel_kong
    fe_fd_e=fe_eva_logmel_kong_fd
    text="logmel_kong"
    return fe_fd,fe_fd_e,text


def feature3():
    fe_fd = fe_logmel_libd_fd   # logmel_librosa_delta
    fe_fd_e=fe_eva_logmel_libd_fd
    text="logmel_lib_delta"
    return fe_fd,fe_fd_e,text

def feature4():
    fe_fd=fe_mel_fd             # mel
    fe_fd_e=fe_eva_mel_fd
    text="mel"
    return fe_fd,fe_fd_e,text
    

def get_dimension(feature):
    
        return {
            "cqt"               :80,
            "logmel_kong"       :40,
            "logmel_lib_delta"  :80,
            "mel"               :40
        }.get(feature, 1000) 

def get_feature(feature):
        return {
            "cqt"                :feature1(),
            "logmel_kong"        :feature2(),
            "logmel_lib_delta"   :feature3(),
            "mel"                :feature4()
        }.get(feature, 1000)
    
def mat_2d_to_3d(X, agg_num, hop):
    # pad to at least one block
    len_X, n_in = X.shape
    if (len_X < agg_num):
        X = np.concatenate((X, np.zeros((agg_num-len_X, n_in))))
        
    # agg 2d to 3d
    len_X = len(X)
    i1 = 0
    X3d = []
    while (i1+agg_num <= len_X):
        X3d.append(X[i1:i1+agg_num])
        i1 += hop
    return np.array(X3d)	