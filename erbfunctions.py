# -*- coding: utf-8 -*-
"""
@author: js
"""

import numpy as np

def frequency2ERBnumber(f):
    return 21.366 * np.log10( 0.004368 * f + 1 )

def ERBnumber2frequency(erb):
    return ( 10 ** ( erb / 21.366 ) - 1 ) / 0.004368

def ERBwidth(f):
    ''' bandwidth of an auditory filter at f (Hz) in Hz
        approximation to ERBnumber2frequency(frequency2ERBnumber(f)+0.5) - ERBnumber2frequency(frequency2ERBnumber(f)-0.5) '''
    return 24.673 *  (0.004368 * f + 1 )