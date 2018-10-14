## Acoustic-Feature-Fusion @Chime18

Code for the paper

### Acoustic Scene Classification
![](https://github.com/DeepLearn-lab/Acoustic-Feature-Fusion_Chime18/blob/master/architecture/tagging%20(1).jpg)

### Audio Tagging
![](https://github.com/DeepLearn-lab/Acoustic-Feature-Fusion_Chime18/blob/master/architecture/tagging.jpg)

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

## Evaluation

## Visualizing Results


| Tables        | Are           | Cool  |
| ------------- |:-------------:| -----:|
| col 3 is      | right-aligned | $1600 |
| col 2 is      | centered      |   $12 |
| zebra stripes | are neat      |    $1 |
