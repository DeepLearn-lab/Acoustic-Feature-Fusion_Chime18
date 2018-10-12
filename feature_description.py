"""
Created on Sat Apr 08 11:48:18 2018

author: @akshitac8
"""

import numpy as np
from scipy import signal
import librosa
import wavio
import scipy

def readwav(path):
    Struct = wavio.read( path )
    wav = Struct.data.astype(float) / np.power(2, Struct.sampwidth*8-1)
    fs = Struct.rate
    return wav, fs

def feature_normalize(feature_data):
    """   
    Input:
    Output:
        
    """
    N = feature_data.shape[0]
    S1 = np.sum(feature_data, axis=0)
    S2 = np.sum(feature_data ** 2, axis=0)
    mean=S1/N
    std=np.sqrt((N * S2 - (S1 * S1)) / (N * (N - 1)))
    mean = np.reshape(mean, [1, -1])
    std = np.reshape(std, [1, -1])
    feature_data=((feature_data-mean)/std)
    return feature_data

def convert_mono(wav,mono):
    """   
    Input:
    Output:
        
    """
    if mono=='mono' and wav.ndim==2:
        return np.mean( wav, axis=-1 )
    if wav.shape[-1]==1 and mono in ['left','right','stereo']:
        raise ValueError("Cannot take channels from mono audio")
    else:
        if mono =='left':
            return wav.T[0]
        elif mono=='right':
            return wav.T[1]
        elif mono=='stereo':
            return wav.T
    return wav  

def read_audio(library,path,fsx):
    """   
    Input: 'str','str','str'
    Output: np.ndarray, int
        
    """
    if library == 'librosa':
        wav,fs = librosa.load(path,sr=fsx)
    elif library == 'readwav':
        wav,fs=readwav(path)
    else:
        raise Exception("Dataset not listed")
    return wav, fs
        
def mel(features,path,library='readwav'):
    
    """
    This function extracts mel-spectrogram from audio.
    Make sure, you pass a dictionary containing all attributes
    and a path to audio.
    """
    fsx=features['fs'][0]
    n_mels=features['n_mels'][0]
    mono=features['mono'][0]
    window_length=features['window_length'][0]
    noverlap=features['noverlap'][0]
    detrend=features['detrend'][0]
    return_onesided=features['return_onesided'][0]
    mode=features['mode'][0]
    normalize=features['normalize'][0]

    wav, fs = read_audio(library,path,fsx)
    wav=convert_mono(wav,mono)
    if fs != fsx:
        raise Exception("Assertion Error. Sampling rate Found {} Expected {}".format(fs,fsx))
    ham_win = np.hamming(window_length)
    [f, t, X] = signal.spectral.spectrogram(wav,fs, window=ham_win, nperseg=window_length, noverlap=noverlap, detrend=detrend, return_onesided=return_onesided, mode=mode )
    X = X.T

    # define global melW, avoid init melW every time, to speed up.
    if globals().get('melW') is None:
        global melW
        melW = librosa.filters.mel( fs, n_fft=window_length, n_mels=n_mels, fmin=0., fmax=fsx/2. )
        melW /= np.max(melW, axis=-1)[:,None]
    
    X = np.dot( X, melW.T )
    X = X[:, 0:]
    if normalize:
        X=feature_normalize(X)
    return X

def logmel(features,path,library='readwav'):
    """
    This function extracts log mel-spectrogram from audio.
    Make sure, you pass a dictionary containing all attributes
    and a path to audio.
    """
    fsx=features['fs'][0]
    n_mels=features['n_mels'][0]
    mono=features['mono'][0]
    window_length=features['window_length'][0]
    noverlap=features['noverlap'][0]
    detrend=features['detrend'][0]
    return_onesided=features['return_onesided'][0]
    mode=features['mode'][0]
    normalize=features['normalize'][0]

    wav, fs = read_audio(library,path,fsx)
    #print "fs before mono",fs #[DEBUG]
    wav=convert_mono(wav,mono)
    if fs != fsx:
        raise Exception("Assertion Error. Sampling rate Found {} Expected {}".format(fs,fsx))
    ham_win = np.hamming(window_length)
    [f, t, X] = signal.spectral.spectrogram(wav,fs, window=ham_win, nperseg=window_length, noverlap=noverlap, detrend=detrend, return_onesided=return_onesided, mode=mode )
    X = X.T

    # define global melW, avoid init melW every time, to speed up.
    if globals().get('melW') is None:
        global melW
        melW = librosa.filters.mel( fs, n_fft=window_length, n_mels=n_mels, fmin=0., fmax=fs/2. )
        melW /= np.max(melW, axis=-1)[:,None]
        #print "mel"
    
    X = np.dot( X, melW.T )
    X = np.log( X + 1e-8)
    X = X[:, 0:]
    
    if normalize:
        X=feature_normalize(X)
    
    return X

def cqt(features,path,library='readwav'):
    """
    This function extracts constant q-transform from audio.
    Make sure, you pass a dictionary containing all attributes
    and a path to audio.
    """    
    fsx = features['fs'][0]
    hop_length = features['hop_length'][0]
    n_bins = features['n_bins'][0]
    bins_per_octave = features['bins_per_octave'][0]
    window_type = features['window_type'][0]
    mono=features['mono'][0]
    normalize=features['normalize'][0]

    wav, fs = read_audio(library,path,fsx)
    wav=convert_mono(wav,mono)
    if fs != fsx:
        raise Exception("Assertion Error. Sampling rate Found {} Expected {}".format(fs,fsx))
    X=librosa.cqt(y=wav, hop_length=hop_length,sr=fs, n_bins=n_bins, bins_per_octave=bins_per_octave,window=window_type)
    X=X.T
    X=np.abs(np.log10(X))
    
    if normalize:
        X=feature_normalize(X)

    return X


#def mfcc(features,path):
def spectralCentroid(features,path,library='readwav'):
    fsx = features['fs'][0]
    mono=features['mono'][0]
#    normalize=features['normalize'][0]

    wav, fs = read_audio(library,path,fsx)
    wav=convert_mono(wav,mono)
    if fs != fsx:
        raise Exception("Assertion Error. Sampling rate Found {} Expected {}".format(fs,fsx))
    spectrum = abs(np.fft.rfft(wav))
    normalized_spectrum = spectrum / sum(spectrum)  # like a probability mass function
    normalized_frequencies = np.linspace(0, 1, len(spectrum))
    X = sum(normalized_frequencies * normalized_spectrum)
    
#    if normalize:
#        X=feature_normalize(X)
         
    return X
    
def zcr(features,path,library='readwav'):
    fsx = features['fs'][0]
    mono=features['mono'][0]
    frame_length = features['frame_length'][0]
    hop_length = features['hop_length'][0]
    center = features['center'][0]
    pad = features['pad'][0]
#    normalize=features['normalize'][0]

    wav, fs = read_audio(library,path,fsx)
    wav=convert_mono(wav,mono)
    if fs != fsx:
        raise Exception("Assertion Error. Sampling rate Found {} Expected {}".format(fs,fsx))
    X=librosa.feature.zero_crossing_rate(wav, frame_length=frame_length, hop_length=hop_length, center=center, pad=pad)
    X=X.T
   
#    if normalize:
#        X=feature_normalize(X)

    return X

def stft(features,path,library='readwav'):
    fsx = features['fs'][0]
    window_length = features['window_length'][0]
    mono=features['mono'][0]
    noverlap=features['noverlap'][0]
    detrend=features['detrend'][0]
    return_onesided=features['return_onesided'][0] 
    boundary = features['boundary'][0]
    padded = features['padded'][0]
    normalize=features['normalize'][0]

    wav, fs = read_audio(library,path,fsx)
    wav=convert_mono(wav,mono)
    if fs != fsx:
        raise Exception("Assertion Error. Sampling rate Found {} Expected {}".format(fs,fsx))
    ham_win = np.hamming(window_length)
    f,t,X = scipy.signal.stft(wav, fs, window=ham_win, nperseg=window_length, noverlap=noverlap, detrend=detrend, return_onesided=return_onesided, boundary=boundary, padded=padded, axis=0)
   
    if normalize:
        X=feature_normalize(X)

    return X    

def SpectralRolloff(features,path,library='readwav'):
    fsx = features['fs'][0]
    mono=features['mono'][0]
    noverlap=features['noverlap'][0]
    detrend=features['detrend'][0]
    return_onesided=features['return_onesided'][0]
    mode=features['mode'][0]
    window_length = features['window_length'][0]
    hop_length = features['hop_length'][0]
    roll_percent = features['roll_percent'][0]
    freq = features['freq'][0]
#    normalize=features['normalize'][0]
    if not freq:
        freq=None
    wav, fs = read_audio(library,path,fsx)
    wav=convert_mono(wav,mono)
    if fs != fsx:
        raise Exception("Assertion Error. Sampling rate Found {} Expected {}".format(fs,fsx))
        
    ham_win = np.hamming(window_length)
    [f, t, x] = signal.spectral.spectrogram(wav,fs, window=ham_win, nperseg=window_length, noverlap=noverlap, detrend=detrend, return_onesided=return_onesided, mode=mode )
    X = librosa.feature.spectral_rolloff(wav, sr=fs, S=x, n_fft=window_length, hop_length=hop_length, freq=freq, roll_percent=roll_percent)
    X = X.T
    
#    if normalize:
#        X=feature_normalize(X)

    return X

def istft(features,path,library='readwav'):
    fsx = features['fs'][0]
    mono=features['mono'][0]
    window_type = features['window_type'][0]
    noverlap=features['noverlap'][0]
    window_length = features['window_length'][0]
    normalize=features['normalize'][0]

  
    wav, fs = read_audio(library,path,fsx)
    wav=convert_mono(wav,mono)
    if fs != fsx:
        raise Exception("Assertion Error. Sampling rate Found {} Expected {}".format(fs,fsx))
    stft_matrix = stft(features,path)
    t, X = scipy.signal.istft(stft_matrix, fs, window='hann', nperseg=None, noverlap=None, nfft=None, input_onesided=True, boundary=True, time_axis=-1, freq_axis=-2)   
    if normalize:
        X=feature_normalize(X)

    return X 
