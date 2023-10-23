#!/bin/python3

import numpy as np
import matplotlib.pyplot as plt

def gav_vector(lat: float, alt:float):

    g0 = 9.7803253359 * (1 + 0.001931853 * np.sin(lat)**2) \
        / np.sqrt(0.00669438 * np.sin(lat)**2)
    
    g = g0 * (1 - (3.157042870579883e-7 - 2.10268965044023439 * np.sin(lat)**2) \
              * alt + 7.374516772941995e-14 * alt**2) 
    
    gav_vec = np.array([[0], [0], [g]])

    return gav_vec

def main():

    sigma_p = 3         # m
    sigma_v = 0.2       # m/s
    sigma_ba = 0.0005   # g*s^1/2
    tau_s = 300         # s
    sigma_wa = 0.12     # g
    sigma_bg = 0.3      # deg/s^1/2
    tau_g = 300         # s
    sigma_wg = 0.95     # deg/s

    acc_bias = np.array([[0.25], [0.077], [-0.12]])
    gyro_bias = np.array([[2.4], [1.3], [5.6]]) * 1e-4

    gps_data = np.genfromtxt('gps.txt', delimiter=',', dtype=float)
    imu_data = np.genfromtxt('imu.txt', delimiter=',', dtype=float)

    start_time = imu_data[0][0]
    end_time = imu_data[-1][0]
    
    pass

    return 1

if __name__=="__main__":
    main()