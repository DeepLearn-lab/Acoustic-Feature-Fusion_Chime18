#Imports
'''
config file to store the variable path and values for tagging task
'''

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