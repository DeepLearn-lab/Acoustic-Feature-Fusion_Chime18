import numpy as np
import cPickle
import os
from scipy import signal
from scikits.audiolab import wavread
import librosa
import utils as cfg
from hat.preprocessing import mat_2d_to_3d,reshape_3d_to_4d
from scipy.fftpack import fft
from scipy.signal  import lfilter, hamming
from scipy.fftpack.realtransforms import dct
from scikits.talkbox import segment_axis
from numpy import abs, sum, linspace
from numpy.fft import rfft

import wavio

#for wavs48 do mean i.e, convert stereo to mono
#for wavs16 dont do mean    
def readwav( path ):
    Struct = wavio.read( path )
    wav = Struct.data.astype(float) / np.power(2, Struct.sampwidth*8-1)
    fs = Struct.rate
    return wav, fs


    
def feature_normalize(feature_data):
    mean = np.mean(feature_data, axis=0)
    std = np.std(feature_data, axis=0)
    N = feature_data.shape[0]
    S1 = np.sum(feature_data, axis=0)
    S2 = np.sum(feature_data ** 2, axis=0)
    mean=S1/N
    std=np.sqrt((N * S2 - (S1 * S1)) / (N * (N - 1)))
    mean = np.reshape(mean, [1, -1])
    std = np.reshape(std, [1, -1])
    feature_data=((feature_data-mean)/std)
    return feature_data
    
     
def cqt_lib(wav_fd,fe_fd):
    names = [ na for na in os.listdir(wav_fd) if na.endswith('.wav') ]
    names = sorted(names)
    for na in names:
        path = wav_fd + '/' + na
        wav, sr = librosa.load( path ,sr=16000.)
        if ( wav.ndim==2 ): 
            wav = np.mean( wav, axis=-1 )
        assert sr==16000.
        cqt=librosa.core.cqt(y=wav, hop_length=512,sr=sr, n_bins=80, bins_per_octave=12, window='hamming')
        cqt=cqt.T
#        cqt=np.log10(cqt)
        cqt=feature_normalize(cqt)
        cqt=np.log10(cqt)
        out_path = fe_fd + '/' + na[0:-4] + '.f'
        cPickle.dump( cqt, open(out_path, 'wb'), protocol=cPickle.HIGHEST_PROTOCOL )
      

##################################################mel kong uncle with 64 coefficients############################################    
    
def mel(wav_fd,fe_fd,n_delete):
    names = [ na for na in os.listdir(wav_fd) if na.endswith('.wav') ]
    names = sorted(names)
    for na in names:
        path = wav_fd + '/' + na
        wav,sr=librosa.load(path,sr=16000.)
        if ( wav.ndim==2 ): 
            wav = np.mean( wav, axis=-1 )
        assert sr==16000.
        ham_win = np.hamming(1024)
        [f, t, X] = signal.spectral.spectrogram( wav, fs=sr,window=ham_win, nperseg=1024, noverlap=0, detrend=False, return_onesided=True, mode='magnitude' ) 
        X = X.T
        if globals().get('melW') is None:
            global melW
            melW = librosa.filters.mel( sr=sr, n_fft=1024, n_mels=60, fmin=0., fmax=sr/2. )
            melW /= np.max(melW, axis=-1)[:,None]
        X = np.dot( X, melW.T )
        X = X[:, n_delete:]
        X=feature_normalize(X)
        out_path = fe_fd + '/' + na[0:-4] + '.f'
        cPickle.dump( X, open(out_path, 'wb'), protocol=cPickle.HIGHEST_PROTOCOL )

  
       
def logmel_lib_delta(wav_fd,fe_fd):
     names = [ na for na in os.listdir(wav_fd) if na.endswith('.wav') ]
     names = sorted(names)
     for na in names:
         path = wav_fd + '/' + na
         wav, sr = librosa.load( path )
         if ( wav.ndim==2 ):                                                              
             wav = np.mean( wav, axis=-1 )
 #        assert fs==44100
         nceps=13
         if True:
             stft = np.abs(librosa.stft(wav, n_fft=1024 , hop_length=512, win_length=1024, window='hamming', center=True))
             MFCCs = np.log(1+librosa.feature.melspectrogram(wav, sr=sr,S=stft,n_mels=40, n_fft=1024, hop_length=512, power=2))
             MFCCs=MFCCs.T
             #MFCCs = dct(MFCCs, type=2, norm='ortho', axis=-1)[:, :nceps]
             mfcc_delta=librosa.feature.delta(MFCCs, width=9, order=1, axis=-1, trim=True)[:, :nceps]
             mfcc_acc=librosa.feature.delta(MFCCs, width=9, order=2, axis=-1, trim=True)[:, :nceps]
             MFCCs=np.hstack((MFCCs,mfcc_delta,mfcc_acc))
             MFCCs-=np.mean(MFCCs)
             MFCCs/=np.std(MFCCs, axis=0)
             print MFCCs.shape
             out_path = fe_fd + '/' + na[0:-4] + '.f'
             cPickle.dump(MFCCs, open(out_path, 'wb'), protocol=cPickle.HIGHEST_PROTOCOL)


    
def logmel_kong(wav_fd,fe_fd,n_delete):
    names = [ na for na in os.listdir(wav_fd) if na.endswith('.wav') ]
    names = sorted(names)
    for na in names:
        path = wav_fd + '/' + na
        wav, sr = librosa.load( path , sr=16000.)
        if ( wav.ndim==2 ): 
            wav = np.mean( wav, axis=-1 )
        assert sr==16000.
        ham_win = np.hamming(1024)
        [f, t, X] = signal.spectral.spectrogram( wav, fs=sr,window=ham_win, nperseg=1024, noverlap=0, detrend=False, return_onesided=True, mode='magnitude' ) 
        X = X.T
        
#        if globals().get('melW') is None:
        global melW
        melW = librosa.filters.mel( sr, n_fft=1024, n_mels=40, fmin=0., fmax=sr/2. )
#        print 'gLU'
        melW /= np.max(melW, axis=-1)[:,None]
            
        X = np.dot( X, melW.T )
        X = np.log(X + 1e-8)
        X = X[:,  n_delete:]
        X=feature_normalize(X) # DONT MOVE THIS ANYWHERE
#        print X.shape
        
        out_path = fe_fd + '/' + na[0:-4] + '.f'
        cPickle.dump( X, open(out_path, 'wb'), protocol=cPickle.HIGHEST_PROTOCOL )

def CreateFolder( fd ):
    if not os.path.exists(fd):
        os.makedirs(fd)
        
def extract(f):
    if f=='mel':
        mel(cfg.wav_fd_train,cfg.fe_mel_fd,n_delete=0)
        mel(cfg.wav_fd_test,cfg.fe_eva_mel_fd,n_delete=0)
    elif f=='cqt':
        cqt_lib(cfg.wav_fd_train,cfg.fe_cqt_fd)
        cqt_lib(cfg.wav_fd_test,cfg.fe_eva_cqt_fd)
    elif f=='logmel_kong':
        logmel_kong(cfg.wav_fd_train,cfg.fe_logmel_kong_fd,n_delete=0)
        logmel_kong(cfg.wav_fd_test,cfg.fe_eva_logmel_kong_fd,n_delete=0)
    elif f=='logmel_lib_delta':
        logmel_lib_delta(cfg.wav_fd_test,cfg.fe_eva_logmel_libd_fd)    
        logmel_lib_delta(cfg.wav_fd_train,cfg.fe_logmel_libd_fd)    
        
if __name__ == "__main__":
    
    
    # CreateFolder('Fe')
    CreateFolder(cfg.fe_cqt_fd)
    CreateFolder(cfg.fe_logmel_libd_fd)   
    CreateFolder(cfg.fe_mel_fd)
    CreateFolder(cfg.fe_logmel_kong_fd)
    
    # CreateFolder('Fe_eva')
    CreateFolder(cfg.fe_eva_cqt_fd)
    CreateFolder(cfg.fe_eva_logmel_libd_fd)   
    CreateFolder(cfg.fe_eva_mel_fd)
    CreateFolder(cfg.fe_eva_logmel_kong_fd)
    
    
    features=['logmel_kong','cqt']
    for f in features:
        extract(f)