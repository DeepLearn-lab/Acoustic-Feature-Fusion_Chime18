# -*- coding: utf-8 -*-
import utils as cfg
import numpy as np
import cPickle
import csv
print("Loading prepare data")
'''
Fetching the feature values
'''
def feature1():
    fe_fd   =  cfg.fe_cqt_fd   # cqt
    fe_fd_e =  cfg.fe_eva_cqt_fd
    text="cqt"
    return fe_fd,fe_fd_e,text

def feature2():
    fe_fd   =  cfg.fe_logmel_kong_fd   # logmel_kong
    fe_fd_e =  cfg.fe_eva_logmel_kong_fd
    text="logmel_kong"
    return fe_fd,fe_fd_e,text


def feature3():
    fe_fd   =  cfg.fe_logmel_libd_fd   # logmel_librosa_delta
    fe_fd_e =  cfg.fe_eva_logmel_libd_fd
    text="logmel_lib_delta"
    return fe_fd,fe_fd_e,text

def feature4():
    fe_fd   = cfg.fe_mel_fd             # mel
    fe_fd_e = cfg.fe_eva_mel_fd
    text="mel"
    return fe_fd,fe_fd_e,text
    

def get_dimension(feature):
    # Given the demensions of the features
    
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

def tGetAllData( fe_fd, eva_csv,agg_num, hop,fetur):
    '''
    Returns Training and Testing Data and Labels
    '''
    with open( eva_csv, 'rb') as g:
        reader2 = csv.reader(g)
        lis2 = list(reader2)

    X3d_all = []
    te_ylist = []
    i=0
    for li in lis2:
        # load data
        na = li[1]            
        path = fe_fd + '/' + na + '.f'           
        info_path = cfg.label_csv + '/' + na + '.csv'
        with open( info_path, 'rb') as g: #Read the Label file to point to the specific labels
            reader2 = csv.reader(g)
            lis2 = list(reader2)
            
        tags = GetTags(info_path, path)
        y = TagsToCategory( tags )
        try:
            X = cPickle.load( open( path, 'rb' ) )
        except Exception as e:
            print 'Error while parsing',path
            continue
        # reshape data to (n_block, n_time, n_freq)
        i+=1
        if i%100==0:
            print "Files Loaded",i
        X3d = mat_2d_to_3d( X, agg_num, hop )
        X3d_all.append( X3d )
        te_ylist += [ y ] * len( X3d )
        #y_all += [ y ]        
    print 'All files loaded successfully'
    # concatenate list to array
    X3d_all = np.concatenate( X3d_all )
    te_ylist = np.array( te_ylist )
       
    return X3d_all, te_ylist

def GetTags( info_path, path):
    # Returns the tags associated with each audio
    with open( info_path, 'rb') as f:
        reader = csv.reader(f)
        lis = list(reader)
    tags = lis[-2][1]
    return tags
            
# tags to categorical, shape: (n_labels)
def TagsToCategory( tags ):
    y = np.zeros( len(cfg.labels) )
    for ch in tags:
        y[ cfg.lb_to_id[ch] ] = 1
    return y