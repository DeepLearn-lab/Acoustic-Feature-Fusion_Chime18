## Acoustic-Feature-Fusion @Chime18

Code for the paper

### Acoustic Scene Classification
![](https://github.com/DeepLearn-lab/Acoustic-Feature-Fusion_Chime18/blob/master/architecture/abc.JPG)

### Audio Tagging
![](https://github.com/DeepLearn-lab/Acoustic-Feature-Fusion_Chime18/blob/master/architecture/tagging.png)

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

## Extracting Audio features
To extract audio features, run the following:

```
python feature_extraction.py

```
The extracted features will get saved at Fe/ and Fe_eva/ folders.

## Training

#### Audio Tagging

Training with defaut parameters works in the following way

```
python mainfile.py

```
You can add your own module in `models.py`

#### Classification


## Evaluation

#### Audio Tagging

```
python eva-tagging.py
```

#### Classification

```
python eva-classify.py

```

## Visualizing Results
 

The Training curves for vanilla model and proposed architecture,

![](https://github.com/DeepLearn-lab/Acoustic-Feature-Fusion_Chime18/blob/master/Results/loss.PNG)

The training loss and Mean Squared Error(MSE) for the attention based system shows a steep decrease in the loss function as compared to the vanilla model.

#### Performance of the proposed model on `Dcase`, `LITIS-Rouen` and `CHiME-Home`.

Techniques | DCASE | ROUEN (F1) | CHiME-Home |
--- | --- | --- | ---
**Vanilla** | 85.50 | 96.85 | 15.6
**EF** | 86.1 | 96.36 | 15.0
**LF** | 87.00 | 96.80 | 14.6
**EF+LF** | **88.70** | **98.25** | **14.0**

## License


