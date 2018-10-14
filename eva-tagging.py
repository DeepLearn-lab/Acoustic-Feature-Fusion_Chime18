# -*- coding: utf-8 -*-

from sklearn import metrics
import numpy as np


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