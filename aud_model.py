"""
Created on Sat Apr 08 11:48:18 2018
author: @akshitac8
"""

import model_description as M
from keras import backend as K
import models as mod
K.set_image_dim_ordering('th')

class Ensemble_Model:
    """
    Class for Ensemble model.
    
    Supported Models
    ----------
    ENSEMBLE_CNN      : Convolution Neural Network
    
    ENSEMBLE_RNN      : Recurrent Neural Network
    
    ENSEMBLE_CRNN     : Convolution Recurrent Neural Network
    

    Parameters
    ----------
    model : str
        Name of Model
    dimx0 : int
        Second Last Dimension
    dimy0 : int
        Last Dimension
    dimx1 : int
        Second Last Dimension
    dimy1 : int
        Last Dimension
    num_classes : int
        Number of Classes
        
    Returns
    -------
    Ensemble Model
    """
    def __init__(self,model,dimx0,dimy0,dimx1,dimy1,dimx2,dimy2,num_classes,**kwargs):
        if model is None:
            raise ValueError("No model passed")
        self.model=model
        self.dimx0 = dimx0
        self.dimy0 = dimy0
        self.dimx1 = dimx1
        self.dimy1 = dimy1
        self.dimx2 = dimx2
        self.dimy2 = dimy2
        self.num_classes=num_classes
        self.kwargs=kwargs

    def prepare_model(self):
        """
        This function
        """
        if self.model == 'ensemble_cnn':
            lrmodel=M.ensemble_cnn(num_classes = self.num_classes, dimx0 = self.dimx0, dimy0 = self.dimy0, dimx1 = self.dimx1, dimy1 = self.dimy1, kwargs=self.kwargs)
        elif self.model == 'ensemble_rnn':
            lrmodel=M.ensemble_rnn(num_classes = self.num_classes, dimx0 = self.dimx0, dimy0 = self.dimy0, dimx1 = self.dimx1, dimy1 = self.dimy1, kwargs=self.kwargs)
        elif self.model == 'ensemble_cnn_rnn':
            lrmodel=M.ensemble_cnn_rnn(num_classes = self.num_classes, dimx0 = self.dimx0, dimy0 = self.dimy0, dimx1 = self.dimx1, dimy1 = self.dimy1, kwargs=self.kwargs)
        elif self.model == 'ensemble':
            lrmodel = mod.ensemble(num_classes = self.num_classes, dimx0 = self.dimx0, dimy0 = self.dimy0, dimx1 = self.dimx1, dimy1 = self.dimy1,dimx2 = self.dimx2, dimy2 = self.dimy2, kwargs=self.kwargs)
        else:
            raise ValueError("Could not find model {}".format(self.model))
        return lrmodel         
                     
