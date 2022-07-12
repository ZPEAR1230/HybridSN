import numpy as np
import matplotlib.pyplot as plt
import random
from math import pi, e, cos

from numpy import dtype
from scipy.ndimage import gaussian_filter1d


def add_target_pixel(r, H, pxl):
    """
    input:
            # curve: background pixel curve, [numpy.ndarray]
            H: depth, [int]
            pxl: pixel, [int]
    return:
            new_curve: 1d npy
    param:
            'r_inf' :   reflectance spectrum of water column (the mean of all_curve)
            'r'     :   sensor-observed spectrum
            'r_b'   :   reflectance spectrum of target (by USGS Library)
            'a'     :   absorption rate (estimate by the IOPE_Net)
            'b'     :   scatter rate (estimate by the IOPE_Net)
            'h'     :   depth of target
            'k_d'   :   downwelling attenuation coefficients
            'k-uc'  :   upwelling attenuation coefficients of water column
            'k-ub'  :   upwelling attenuation coefficients of target column
    """
    a = np.load(r'D:\ZPEAR\Experiment_data\HybridSN\a.npy')
    b = np.load(r'D:\ZPEAR\Experiment_data\HybridSN\bb.npy')
    r_b = r
    a = np.array(a)
    b = np.array(b)
    r_b = np.array(r_b)
    theta = 0
    h = H
    # calculate underwater-target reflectance
    u = b / (a + b)
    k = a + b

    HSI_curve = np.load(r"D:\ZPEAR\Experiment_data\SUTDF\data\Simulation_data\all_curve.npy")
    # print(a.shape)
    # print(b.shape)
    # print(r_b.shape)
    # print(HSI_curve[4].shape)
    # print(type(a))
    # print(type(b))
    # print(type(r_b))
    # print(type(HSI_curve[4]))

    r_inf = np.array(HSI_curve[pxl])  # 真实的无限水深反射率
    k_uc = 1.03 * (1 + 2.4 * u) ** (0.5) * k
    k_ub = 1.04 * (1 + 5.4 * u) ** (0.5) * k
    k_d = k / cos(theta)
    k_uc = k_uc
    k_ub = k_ub
    k_d = k_d
    r_inf = r_inf
    r = r_inf * (1 - e ** (-(k_d + k_uc) * h)) + r_b / pi * e ** (-(k_d + k_ub) * h)

    # add Gauss noise
    # mu = 0
    # sigma = 0.001
    # for i in range(r.size):
    #         r[i] += random.gauss(mu, sigma)

    return r
