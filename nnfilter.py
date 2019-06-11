# -*- coding: utf-8 -*-
"""
@author: js2251
"""

import numpy as np
import tensorflow as tf

from erbfunctions import ERBnumber2frequency
from scipy.signal import firwin
import matplotlib.pyplot as plt


def generateBandpassFirErb( erb, erb_width=1, num_taps = 2048, window='hamming', fs = 32000, **kwargs ):
    ''' return filter coefficients for bandpass filter, centered at ERB-number
        erb and a width of erb_width, both specified in the ERB-number scale
        (Unit: Cam)'''
    f_lo = ERBnumber2frequency( erb - 0.5 * erb_width )
    f_hi = ERBnumber2frequency( erb + 0.5 * erb_width )
    h    = firwin( num_taps, [f_lo,f_hi],window=window,fs=fs, pass_zero=False,**kwargs )
    return h

def generateKernel( erb_lo = 1.75, num_bands = 150, step_size = 0.25, num_taps = 2048, fs = 32000 ):
    k = np.empty(shape=(num_taps,1,num_bands),dtype='float32')
    for i in range(num_bands):
        k[:,0,i] = generateBandpassFirErb( erb_lo+i*step_size, erb_width=1, num_taps = num_taps, fs = fs  )
    return k
        
def generateBandLevelModel( erb_lo = 1.75, num_bands = 150, step_size = 0.25, num_taps = 2048, fs = 32000, spectra_per_second = 1000 ):
    ''' genrate a neural net that transforms a waveform in ERB-filter levels (dB relative 1 in the waveform)
        inputs: first frequency on ERB-number scale, number of ERB bands and steps size between
        nunmber of taps for FIR filter, sampling frequency, and how many ERB spectra to compute per second '''
    k = generateKernel(erb_lo,num_bands,step_size,num_taps,fs)
    model = tf.keras.models.Sequential()
    model.add( tf.keras.layers.Conv1D( num_bands, kernel_size=(num_taps), kernel_initializer = tf.constant_initializer( k ) ) )
    model.add( tf.keras.layers.Lambda(lambda x: tf.square(x)) )
    model.add( tf.keras.layers.AveragePooling1D( pool_size = num_taps, strides = int(fs/1000) ) )
    model.add( tf.keras.layers.Lambda(lambda x: 10 * tf.log(x)/tf.log(10.0) ) )
    model.trainable = False
    return model


### unit tests

def testBandpass():    
    ''' test for generateBandpassFirErb() '''
    fs = 32000
    t  = list(range(2048))
    s1 = np.sin( np.multiply(t, 2 * np.pi / fs * 1000) )
    s2 = np.sin( np.multiply(t, 2 * np.pi / fs * 800) )
    s3 = np.sin( np.multiply(t, 2 * np.pi / fs * 100) )
    h  = generateBandpassFirErb( 15.6 )   
    s1 = np.convolve( h,s1 ) 
    s2 = np.convolve( h,s2 )
    s3 = np.convolve( h,s3 )

    print('max values: ' + str(np.max(s1)) + ' ' + str(np.max(s2)) + ' ' + str(np.max(s3)) + ' ')   
    plt.plot(list(range(2048)),h)
    
    return s1,s2,s3


def testModel():
    ''' test generateBandLevelModel() for an FM input '''
    fs = 32000
    model = generateBandLevelModel(num_taps=513)
    #tf.keras.utils.plot_model(model, to_file='model.png', show_shapes=True) # add input_shape in first layer
    
    t    = list(range(64000))
    s    = np.cos( np.multiply(t, 2 * np.pi / fs * 4050 ) + 2000 * np.cos( np.multiply(t, 2*np.pi/fs*2 ) ) )
    s_in = s.reshape(1,s.shape[0],1)
    
    out = model.predict(s_in)
    out = out.reshape(out.shape[1],out.shape[2])
    fig = plt.figure()
    ax = fig.add_axes([0.15, 0.1, 1, 1])
    
    ax.set_xlabel('time/s')
    ax.set_ylabel('ERB-number/Cam')
    ax.pcolor(np.transpose(out))