# -*- coding: utf-8 -*-
import warnings
warnings.filterwarnings("ignore")
import numpy as np
import models as mod
import utils as cfg
import keras.backend as K
K.set_image_dim_ordering('tf')
from keras.callbacks import EarlyStopping, ModelCheckpoint,TensorBoard
from sklearn import metrics
import os
import tensorflow as tf
import prepare as p
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID" 
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
import keras.backend.tensorflow_backend as KTF
gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.1)
sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))
KTF.set_session(sess)
np.random.seed(1234)
print('imported')
num_classes=len(cfg.labels)



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
    n1[i] = p.get_dimension(fetur)
    fe_fd[i],fe_fd_e[i],text[i]=p.get_feature(fetur)
    tr_X[i], tr_y[i] = p.tGetAllData( fe_fd[i], cfg.meta_train_csv, cfg.agg_num, cfg.hop ,fetur)
    te_X[i],te_y[i] = p.tGetAllData( fe_fd_e[i], cfg.meta_test_csv, cfg.agg_num, cfg.hop ,fetur)
    
    
####################################### Reshapes and fetching training and testing Data #############################

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

X0=te_X0
X1=te_X1
X2=te_X2

'''
1. Reshaped first dimension of the array for mapping to happen correctly between each feature.
2. Reshaped last dimension of the array for every feature to have same frame size for the model 
   to concatenate over the feature axis to generate an array of shape (15,15) // (agg_num,agg_num) 
'''
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


lrmodel = mod.ensemble(num_classes = num_classes, dimx0 = dimx0, dimy0 = dimy0, dimx1 = dimx1, dimy1 = dimy1,dimx2 = dimx2, dimy2 = dimy2)
                   
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