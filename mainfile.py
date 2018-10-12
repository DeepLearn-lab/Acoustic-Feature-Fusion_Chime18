# -*- coding: utf-8 -*-
import warnings
warnings.filterwarnings("ignore")
import aud_model as A
import numpy as np
import csv
import cPickle
np.random.seed(1234)
import utils as cfg
import keras.backend as K
K.set_image_dim_ordering('tf')
from keras.callbacks import EarlyStopping, ModelCheckpoint,TensorBoard
from sklearn import metrics
import os
import tensorflow as tf
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID" 
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
import keras.backend.tensorflow_backend as KTF
gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.1)
sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))
KTF.set_session(sess)

print('imported')
num_classes=len(cfg.labels)

def tGetAllData( fe_fd, eva_csv,agg_num, hop,fetur):
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
        with open( info_path, 'rb') as g:
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
        X3d = cfg.mat_2d_to_3d( X, agg_num, hop )
        X3d_all.append( X3d )
        te_ylist += [ y ] * len( X3d )
        #y_all += [ y ]        
    print 'All files loaded successfully'
    # concatenate list to array
    X3d_all = np.concatenate( X3d_all )
    te_ylist = np.array( te_ylist )
       
    return X3d_all, te_ylist

def GetTags( info_path, path):
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

n1          = [0,0,0]
fe_fd       = [0,0,0]
fe_fd_e     = [0,0,0]
text        = [0,0,0]
tr_X        = [0,0,0]
tr_y        = [0,0,0]
te_X        = [0,0,0]
te_y        = [0,0,0]


i=-1 #Init with 0 indexing
for fetur in cfg.feature:
    print fetur
    i+=1
    n1[i] = cfg.get_dimension(fetur)
    #print i
    fe_fd[i],fe_fd_e[i],text[i]=cfg.get_feature(fetur)
    tr_X[i], tr_y[i] = tGetAllData( fe_fd[i], cfg.meta_train_csv, cfg.agg_num, cfg.hop ,fetur)
    te_X[i],te_y[i] = tGetAllData( fe_fd_e[i], cfg.meta_test_csv, cfg.agg_num, cfg.hop ,fetur)
    
    
####################################### Reshapes and fetching training and testing Data #############################
fe_fd_e0=fe_fd_e[0]
fe_fd_e1=fe_fd_e[1]
fe_fd_e2=fe_fd_e[2]

tr_X0=np.array(tr_X[0])
tr_X1=np.array(tr_X[1])
tr_X2=np.array(tr_X[2])

tr_y0=np.array(tr_y[0])
tr_y1=np.array(tr_y[1])
tr_y2=np.array(tr_y[2])

te_X0=np.array(te_X[0])
te_X1=np.array(te_X[1])
te_X2=np.array(te_X[2])

te_y0=np.array(te_y[0])
te_y1=np.array(te_y[1])
te_y2=np.array(te_y[2])

n10=n1[0]
n11=n1[1]
n12=n1[2]

text0=text[0]
text1=text[1]
text2=text[2]

X0=te_X0
X1=te_X1
X2=te_X2


new_teX1=np.zeros([8292, 15, 40])
j=0
for i in range(len(X1)):
    new_teX1[j]=X1[i]
    new_teX1[j+1]=X1[i]
    j+=2    
X1=new_teX1 

X1= X1.reshape((-1,15,80)) 
new_teX1=np.zeros([8292, 15, 80])
j=0
for i in range(len(X1)):
    new_teX1[j]=X1[i]
    new_teX1[j+1]=X1[i]
    j+=2    
X1=new_teX1 

new_teX0=np.zeros([8292, 15, 40])
j=0
for i in range(len(X0)):
    new_teX0[j]=X0[i]
    new_teX0[j+1]=X0[i]
    j+=2    
X0=new_teX0 

X0= X0.reshape((-1,15,80)) 
new_teX0=np.zeros([8292, 15, 80])
j=0
for i in range(len(X0)):
    new_teX0[j]=X0[i]
    new_teX0[j+1]=X0[i]
    j+=2    
X0=new_teX0 

new_trX0=np.zeros([19020,15, 40])
j=0
for i in range(len(tr_X0)):
    new_trX0[j]=tr_X0[i]
    new_trX0[j+1]=tr_X0[i]
    j+=2

tr_X0=new_trX0

tr_X0=tr_X0.reshape((-1,15,80))

new_trX0=np.zeros([19020,15, 80])
j=0
for i in range(len(tr_X0)):
    new_trX0[j]=tr_X0[i]
    new_trX0[j+1]=tr_X0[i]
    j+=2

tr_X0=new_trX0


new_trX1=np.zeros([19020,15, 40])
j=0
for i in range(len(tr_X1)):
    new_trX1[j]=tr_X1[i]
    new_trX1[j+1]=tr_X1[i]
    j+=2

tr_X1=new_trX1

tr_X1= tr_X1.reshape((-1,15,80))

new_trX1=np.zeros([19020,15, 80])
j=0
for i in range(len(tr_X1)):
    new_trX1[j]=tr_X1[i]
    new_trX1[j+1]=tr_X1[i]
    j+=2

tr_X1=new_trX1



dimx0=tr_X0.shape[-2]
dimx1=tr_X1.shape[-2]
dimx2=tr_X2.shape[-2]
dimy0=tr_X0.shape[-1]
dimy1=tr_X1.shape[-1]
dimy2=tr_X2.shape[-1]

###################################################################################################################


md = A.Ensemble_Model(input_neurons = 600, act3 = 'sigmoid', act4 ='relu',
                      model=cfg.model,num_classes=num_classes,
                      dimx0=dimx0,dimy0=dimy0,dimx1=dimx1,dimy1=dimy1,dimx2=dimx2,dimy2=dimy2)
                    
lrmodel=md.prepare_model()
tbcallbacks = [EarlyStopping(monitor='val_mean_squared_error', patience=3),
             ModelCheckpoint(filepath='best_model3.h5', monitor='val_mean_squared_error', save_best_only=True),TensorBoard(log_dir='./logs', write_graph=True, write_images=True)]

lrmodel.fit([tr_X0,tr_X1,tr_X2],tr_y2,batch_size=100,epochs=20,verbose=2,
            validation_data=([X0,X1,X2],te_y2),callbacks=tbcallbacks)


#######################################EER calculation ###################################################
def EER(gt,pred):
    fpr, tpr, thresholds = metrics.roc_curve(gt, pred, drop_intermediate=True)
    eps = 1E-6
    Points = [(0,0)]+zip(fpr, tpr)
    for i, point in enumerate(Points):
        if point[0]+eps >= 1-point[1]:
            break
    P1 = Points[i-1]; P2 = Points[i]
        
    #Interpolate between P1 and P2
    if abs(P2[0]-P1[0]) < eps:
        EER = P1[0]        
    else:        
        m = (P2[1]-P1[1]) / (P2[0]-P1[0])
        o = P1[1] - m * P1[0]
        EER = (1-o) / (1+m)
   
    return EER
 
'''
def prediction(y_pred):
    temp=[0]*8
    thres0 = 0.008
    thres1 = 7
    h={}
    for i in range(8):
        h[i] =[]
    for i,p in enumerate(y_pred):
        for j,q in enumerate(p):
            if (q > thres0):
                h[j].append(q)
    for key in h.keys():
        if len(h[key])>thres1:
            temp[key] = 1
    return temp
'''
y=[]
y_pred_new = []
class_eer=[]
p_y_pred = lrmodel.predict([X0,X1,X2]) # probability, size: (n_block,label)

for i in range(0,len(p_y_pred),12):
    y_pred = np.mean(p_y_pred[i:i+12],axis=0)
    y_pred_new.append(y_pred)
    y.append(te_y2[i].tolist())
y_pred_new, y = np.array(y_pred_new), np.array(y)
print y_pred_new.shape
print y.shape

n_out = y.shape[1]
print y_pred_new.shape
print y.shape
for k in xrange(n_out):
    eer = EER(y[:, k],y_pred_new[:, k])
    print "Class ",k,'ERR ',eer
    class_eer.append(eer)
    

EER1=np.mean(class_eer)
print("EER",EER1)