#cython: boundscheck=False
#cython: wraparound=False
#cython: cdivision=True
#cython: embedsignature=True

import numpy as np
from libc.math cimport floor
# ==============================================================
def hist3D(weight, ax1_data, ax2_data, ax3_data, ax1_info=None, ax2_info=None, ax3_info=None):
    """
    Inputs
    ======
        weight : numpy array
            
        
        ax1_data : numpy array
            
        
        ax2_data : numpy array
            
        
        ax3_data : numpy array
            
        ax1_info : dict
            dictionary keys:
                + min
                + max
                + nbins
    This is a 3D histogram of one dataset (as compared to the joint 
    histogram that is in hist2D)
    """
    #=== typed variables =======================================================
    cdef ssize_t  i, j, k
    cdef ssize_t  ax1_bin, ax2_bin, ax3_bin
    cdef ssize_t  ax1_nbins, ax2_nbins, ax3_nbins
    cdef float  ax1_min, ax1_max, ax1_size
    cdef float  ax2_min, ax2_max, ax2_size
    cdef float  ax3_min, ax3_max, ax3_size
    cdef int  bin_exists
    #=== typed arrays ==========================================================
    cdef float[::1] ax1 = np.ravel(ax1_data, order='C').astype('float32')
    cdef float[::1] ax2 = np.ravel(ax2_data, order='C').astype('float32')
    cdef float[::1] ax3 = np.ravel(ax3_data, order='C').astype('float32')
    cdef float[::1] arr = np.ravel(weight, order='C').astype('float32')
    cdef double[:,:,:] hist
    #=== Make an info dictionary for ax1 and set values from it ================
    if ax1_info == None:
        ax1_info = dict()
        ax1_info['min'] = np.amin(ax1_data)
        ax1_info['max'] = np.amax(ax1_data)
        ax1_info['nbins'] = 201
    else:
        if 'min' not in ax1_info.keys():
            ax1_info['min'] = np.amin(ax1_data)
        if 'max' not in ax1_info.keys():
            ax1_info['max'] = np.amax(ax1_data)
        if 'nbins' not in ax1_info.keys():
            ax1_info['nbins'] = 201
    
    ax1_size = (ax1_info['max'] - ax1_info['min']) / (ax1_info['nbins'] - 1.0)
    ax1_min = ax1_info['min'] - 0.5 * ax1_size
    ax1_max = ax1_info['max'] + 0.5 * ax1_size
    ax1_nbins = ax1_info['nbins']
    #=== Make an info dictionary for ax2 and set values from it ================
    if ax2_info == None:
        ax2_info = dict()
        ax2_info['min'] = np.amin(ax2_data)
        ax2_info['max'] = np.amax(ax2_data)
        ax2_info['nbins'] = 201
    else:
        if 'min' not in ax2_info.keys():
            ax2_info['min'] = np.amin(ax2_data)
        if 'max' not in ax2_info.keys():
            ax2_info['max'] = np.amax(ax2_data)
        if 'nbins' not in ax2_info.keys():
            ax2_info['nbins'] = 201
    
    ax2_size = (ax2_info['max'] - ax2_info['min']) / (ax2_info['nbins'] - 1.0)
    ax2_min = ax2_info['min'] - 0.5 * ax2_size
    ax2_max = ax2_info['max'] + 0.5 * ax2_size
    ax2_nbins = ax2_info['nbins']
    #=== Make an info dictionary for ax3 and set values from it ================
    if ax3_info == None:
        ax3_info = dict()
        ax3_info['min'] = np.amin(ax3_data)
        ax3_info['max'] = np.amax(ax3_data)
        ax3_info['nbins'] = 201
    else:
        if 'min' not in ax3_info.keys():
            ax3_info['min'] = np.amin(ax3_data)
        if 'max' not in ax3_info.keys():
            ax3_info['max'] = np.amax(ax3_data)
        if 'nbins' not in ax3_info.keys():
            ax3_info['nbins'] = 201
    
    ax3_size = (ax3_info['max'] - ax3_info['min']) / (ax3_info['nbins'] - 1.0)
    ax3_min = ax3_info['min'] - 0.5 * ax3_size
    ax3_max = ax3_info['max'] + 0.5 * ax3_size
    ax3_nbins = ax3_info['nbins']
    #=== Make histogram array and fill it ======================================
    hist = np.zeros((ax1_nbins, ax2_nbins, ax3_nbins), dtype='float64')
    
    for i in range(ax1.size):
        ax1_bin = <ssize_t> floor( (ax1[i] - ax1_min) / ax1_size )
        ax2_bin = <ssize_t> floor( (ax2[i] - ax2_min) / ax2_size )
        ax3_bin = <ssize_t> floor( (ax3[i] - ax3_min) / ax3_size )
        # 
        bin_exists = 1
        if ax1_bin < 0: bin_exists = 0
        if ax2_bin < 0: bin_exists = 0
        if ax3_bin < 0: bin_exists = 0
        if ax1_bin >= ax1_nbins: bin_exists = 0
        if ax2_bin >= ax2_nbins: bin_exists = 0
        if ax3_bin >= ax3_nbins: bin_exists = 0
        if bin_exists == 1:
            hist[ax1_bin, ax2_bin, ax3_bin] += arr[i]
    # ==========================================================
    result_dict = {'data': hist, 'ax1_info': ax1_info, 'ax2_info': ax2_info, 'ax3_info': ax3_info}
    return result_dict
# ==============================================================
