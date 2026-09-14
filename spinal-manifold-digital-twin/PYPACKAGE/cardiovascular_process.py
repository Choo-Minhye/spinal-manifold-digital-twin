from scipy import io
import numpy as np
import scipy.signal as sig

from PYPACKAGE.spike_process import gaussian_wind_fn, gaussian_filter

def time_edge_2_cardio_idx(edges):
    cardio_idx = 100*edges
    cardio_idx = np.round((cardio_idx[1:] + cardio_idx[:-1])*0.5).astype(int)
    return cardio_idx

def get_bp(file_path_name, idx = np.array([]), filt = False):
    # get cardiovascualr data
    bp_raw = io.loadmat(file_path_name)
    # bp_raw = io.loadmat('..\\PROCESSED\\bp_data_test_v7.mat')
    bp_raw = bp_raw['BP']
    bp_raw = bp_raw[:,1]
    
    if filt != 0:
        L = 15
        sigma = (idx[1] - idx[0])*0.01
        bp_raw = gaussian_filter(L, sigma, sigma, bp_raw.transpose())
        bp_raw = bp_raw.transpose()
    
    if len(idx) != 0:
        bp_data = bp_raw[idx]
    else:
        bp_data = bp_raw
    return bp_data

import matplotlib.pyplot as plt

def get_bp_with_time(file_path_name) :
    
    bp_t_raw = io.loadmat(file_path_name)    
    bp_raw = bp_t_raw['bp_resampled_']    
    bp_data = bp_raw[0,:]
    
    t_raw = bp_t_raw['t']
    t = t_raw[0,:]
    
    fig = plt.figure() 
    ax = fig.add_subplot(111)
    ax.plot(t,bp_data)
    # plt.show()
    
    return bp_data, t
    
    
