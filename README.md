## Acoustic-Feature-Fusion @Chime18

Code for the [paper](https://www.isca-speech.org/archive/CHiME_2018/abstracts/CHiME_2018_paper_bhatt.html) accepted at <b>Interspeech-Chime' 2018</b>

![](https://github.com/DeepLearn-lab/Acoustic-Feature-Fusion_Chime18/blob/master/architecture/img.png)

## Setup and Dependencies
Our code is implemented in Keras (v2.0.8 with CUDA). To setup, do the following:

If you do not have any Anaconda or Miniconda distribution, head over to their [downloads' site](https://conda.io/docs/user-guide/install/download.html) before proceeding further.

Clone the repository and create an environment.

```
git clone https://github.com/DeepLearn-lab/Acoustic-Feature-Fusion_Chime18.git
conda env create -f environment.yaml
```

This creates an environment named `kerasenv2` with all the dependencies installed.


## Download the Dataset
Download the [CHiME-Home](http://www.cs.tut.fi/sgn/arg/dcase2016/task-audio-tagging), [Acoustic Scene](http://www.cs.tut.fi/sgn/arg/dcase2016/task-acoustic-scene-classification) and [LITIS-Rouen](https://sites.google.com/site/alainrakotomamonjy/home/audio-scene) datasets and unzip the required datasets.

## Code Usage

In the following steps I am going to show how to run the code on the given dataset. (This is what you would need to do for your own audio classification and tagging dataset as well.) The scripts that I am going to mention in each step, please open them and see if you need to the paths.

## Extracting Audio features
To extract audio features, run the following:

```
python feature_extraction.py

```
The extracted features will get saved at Fe/ and Fe_eva/ folders.

## Training

Training with defaut parameters works in the following way

```
1. Audio Tagging
- python mainfile.py

2. Classification
- python mainfile_classify.py

```

You can add your own module in `models.py`

## Evaluation

1. Audio Tagging

```
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

Class  0 ERR  0.14377162629757783
Class  1 ERR  0.20401913875598086
Class  2 ERR  0.2398412698412699
Class  3 ERR  0.12727272727272726
Class  4 ERR  0.1588235294117647
Class  5 ERR  0.06666666666666667
Class  6 ERR  0.03994082840236687
('EER', 0.14019082666405057)
```

2. Classification

```
y=[]
y_pred_new = []
p_y_pred = lrmodel.predict([X0,X1,X2]) # probability, size: (n_block,label)
for i in range(0,len(p_y_pred),258):
    y_pred = np.mean(p_y_pred[i:i+258],axis=0)
    y_pred_new.append(y_pred)
    y.append(te_y2[i].tolist())
y_pred_new, y = np.array(y_pred_new), np.array(y)
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

```

## Visualizing Results
 

We have integrated Tensorboard_logger to visualize training and mean squared error. To install tensorboard logger use :
```
pip install tensorboard_logger
```

The Training curves for vanilla model and proposed architecture for Tagging Task,

![](https://github.com/DeepLearn-lab/Acoustic-Feature-Fusion_Chime18/blob/master/Results/loss.PNG)

The training loss and Mean Squared Error(MSE) for the attention based system shows a steep decrease in the loss function as compared to the vanilla model.

#### Performance of the proposed model on `Dcase`, `LITIS-Rouen` and `CHiME-Home`.

Techniques | DCASE | ROUEN (F1) | CHiME-Home |
--- | --- | --- | ---
**Vanilla** | 85.50 | 96.85 | 15.6
**EF** | 86.1 | 96.36 | 15.0
**LF** | 87.00 | 96.80 | 14.6
**EF+LF** | **88.70** | **98.25** | **14.0**

## Contributors

[![](https://sourcerer.io/fame/akshitac8/DeepLearn-lab/Acoustic-Feature-Fusion_Chime18/images/0)](https://sourcerer.io/fame/akshitac8/DeepLearn-lab/Acoustic-Feature-Fusion_Chime18/links/0)[![](https://sourcerer.io/fame/akshitac8/DeepLearn-lab/Acoustic-Feature-Fusion_Chime18/images/1)](https://sourcerer.io/fame/akshitac8/DeepLearn-lab/Acoustic-Feature-Fusion_Chime18/links/1)[![](https://sourcerer.io/fame/akshitac8/DeepLearn-lab/Acoustic-Feature-Fusion_Chime18/images/2)](https://sourcerer.io/fame/akshitac8/DeepLearn-lab/Acoustic-Feature-Fusion_Chime18/links/2)[![](https://sourcerer.io/fame/akshitac8/DeepLearn-lab/Acoustic-Feature-Fusion_Chime18/images/3)](https://sourcerer.io/fame/akshitac8/DeepLearn-lab/Acoustic-Feature-Fusion_Chime18/links/3)[![](https://sourcerer.io/fame/akshitac8/DeepLearn-lab/Acoustic-Feature-Fusion_Chime18/images/4)](https://sourcerer.io/fame/akshitac8/DeepLearn-lab/Acoustic-Feature-Fusion_Chime18/links/4)[![](https://sourcerer.io/fame/akshitac8/DeepLearn-lab/Acoustic-Feature-Fusion_Chime18/images/5)](https://sourcerer.io/fame/akshitac8/DeepLearn-lab/Acoustic-Feature-Fusion_Chime18/links/5)[![](https://sourcerer.io/fame/akshitac8/DeepLearn-lab/Acoustic-Feature-Fusion_Chime18/images/6)](https://sourcerer.io/fame/akshitac8/DeepLearn-lab/Acoustic-Feature-Fusion_Chime18/links/6)[![](https://sourcerer.io/fame/akshitac8/DeepLearn-lab/Acoustic-Feature-Fusion_Chime18/images/7)](https://sourcerer.io/fame/akshitac8/DeepLearn-lab/Acoustic-Feature-Fusion_Chime18/links/7)

## License
BSD
