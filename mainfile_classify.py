# -*- coding: utf-8 -*-

# -*- coding: utf-8 -*-
import warnings
warnings.filterwarnings("ignore")
import numpy as np
import models as mod
import utils_classify as cfg
import keras.backend as K
K.set_image_dim_ordering('tf')
from keras.callbacks import EarlyStopping, ModelCheckpoint,TensorBoard
import os
import tensorflow as tf
import prepare as p
import scipy
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID" 
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
import keras.backend.tensorflow_backend as KTF
gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.1)
sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))
KTF.set_session(sess)
np.random.seed(1234)
print('imported')
num_classes=len(cfg.labels)
from sklearn import metrics

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
     # Fetching training and testing data of all the features sequentially
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
tr_y2=np.array(tr_y[2])

te_X0=np.array(te_X[0])
te_X1=np.array(te_X[1])
te_X2=np.array(te_X[2])
te_y2=np.array(te_y[2])

X0=te_X0
X1=te_X1
X2=te_X2

'''
1. Reshaped first dimension of the array to accordingly map feature frame.
2. Reshaped last dimension of the array for every feature to have same frame size for the model 
   to concatenate over the feature axis to generate an array of shape (15,15) // (agg_num,agg_num) 
'''
new_trX0=np.zeros([301860,10, 40,1])
j=0
for i in range(len(tr_X0)):
    new_trX0[j]=tr_X0[i]
    new_trX0[j+1]=tr_X0[i]
    j+=2

new_trX2=np.zeros([301860,10, 60, 1])
j=0
for i in range(len(tr_X2)):
    new_trX2[j]=tr_X2[i]
    new_trX2[j+1]=tr_X2[i]
    j+=2

tr_X0=new_trX0
tr_X2=new_trX2

## testing
new_teX0=np.zeros([258,10, 40, 1])
j=0
for i in range(len(X0)):
    new_teX0[j]=X0[i]
    new_teX0[j+1]=X0[i]
    j+=2

new_teX2=np.zeros([258, 10, 60, 1])
j=0
for i in range(len(X2)):
    new_teX2[j]=X2[i]
    new_teX2[j+1]=X2[i]
    j+=2

X0=new_teX0
X2=new_teX2

dimx0=tr_X0.shape[-3]
dimx1=tr_X1.shape[-3]
dimx2=tr_X2.shape[-3]
dimy0=tr_X0.shape[-2]
dimy1=tr_X1.shape[-2]
dimy2=tr_X2.shape[-2]



lrmodel = mod.ensemble(num_classes = num_classes, dimx0 = dimx0, dimy0 = dimy0, dimx1 = dimx1, dimy1 = dimy1,dimx2 = dimx2, dimy2 = dimy2)
                   
tbcallbacks = [EarlyStopping(monitor='val_mean_squared_error', patience=3),
             ModelCheckpoint(filepath='model.h5', monitor='val_mean_squared_error', save_best_only=True),TensorBoard(log_dir='./logs', write_graph=True, write_images=True)]

lrmodel.fit([tr_X0,tr_X1,tr_X2],tr_y2,batch_size=150,epochs=20,verbose=2,
            validation_data=([X0,X1,X2],te_y2),callbacks=tbcallbacks)

lrmodel.save('model.h5')

'''
Evaluation Script
'''

y=[]
y_pred_new = []
class_eer=[]
p_y_pred = lrmodel.predict([X0,X1,X2]) # probability, size: (n_block,label)

for i in range(0,len(p_y_pred),258):
    y_pred = np.mean(p_y_pred[i:i+258],axis=0)
    y_pred_new.append(y_pred)
    y.append(te_y2[i].tolist())
y_pred_new, y = np.array(y_pred_new), np.array(y)
print y_pred_new.shape
print y.shape


preds = np.argmax( y_pred_new, axis=-1 )     # size: (n_block)
b = scipy.stats.mode(preds)  # Finding the maximum value out of the array for predicting the label
pred = int( b[0] )
truth = open(cfg.eval_txt,'r').readlines()
pred.sort()
truth.sort()
pred = [i.split('\t')[1].split('\n')[0]for i in pred]
truth = [i.split('\t')[1]for i in truth]
met = metrics.classification_report(truth,pred)
val = met.split('\n')

print '\n\n'
for i in val:
    if len(i)>0:
        print i

print '\n\n'
pos,neg=0,0 
for i in range(0,len(pred)):
    if pred[i] == truth[i]:
        pos = pos+1
    else:
        neg = neg+1
print "\n\n ********************* Accuracy ********************* \n"

print "correctly classified     --> ",pos
print "not correctly classified -->",neg
print "percentage(%) accuracy   --> ",(float(pos)/float(len(pred)))*100
acc=(float(pos)/float(len(pred)))*100

print(pred)
print(truth)
print(acc)
