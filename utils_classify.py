# -*- coding: utf-8 -*-

a='/akshitadatae6'
b='/akshitadatae9'
c='/akshitadatae2'
wav_fd = 'audiofiles/TUT-acoustic-scenes-2016-development/audio'




fe_cqt_fd 			 = b+'/Fe/cqt' #2
fe_logmel_kong_fd 	 = c+'/Fe/logmel_kong'#4
fe_logmel_libd_fd 	 = b+'/Fe/logmel_lib_delta'#7
fe_mel_fd 			 = a+'/Fe/mel'#9


meta_csv      = 'G:/akshita_workspace/aditya/akshitaimp/dev/meta.txt'
meta_csv_eval = 'G:/akshita_workspace/aditya/akshitaimp/va/meta.txt'
txt_eva_path  = 'G:/akshita_workspace/aditya/akshitaimp/va/evaluation_setup/test.txt'
eval_txt      = 'G:/akshita_workspace/aditya/akshitaimp/va/evaluation_setup/evaluate.txt'

fe_cqt_eva_fd 			 = b+'/Fe_eva/cqt'
fe_logmel_kong_eva_fd 	 = c+'/Fe_eva/logmel_kong'
fe_logmel_libd_eva_fd 	 = b+'/Fe_eva/logmel_lib_delta'
fe_mel_eva_fd 			 = a+'/Fe_eva/mel'

# 1 of 15 acoustic label
labels = [ 'bus', 'cafe/restaurant', 'car', 'city_center', 'forest_path', 'grocery_store', 'home', 'beach', 
            'library', 'metro_station', 'office', 'residential_area', 'train', 'tram', 'park' ]
            
lb_to_id = { lb:id for id, lb in enumerate(labels) }
id_to_lb = { id:lb for id, lb in enumerate(labels) }
