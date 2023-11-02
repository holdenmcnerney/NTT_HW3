#!/bin/python3

from dataclasses import dataclass
import numpy as np
import matplotlib.pyplot as plt
import navpy as nav

def gav_vector(lat: float, alt:float):

    g0 = 9.7803253359 * (1 + 0.001931853 * np.sin(lat)**2) \
        / np.sqrt(1 - 0.00669438 * np.sin(lat)**2)
    
    g = g0 * (1 - (3.157042870579883e-7 - 2.10268965044023439e9 * np.sin(lat)**2)\
              * alt + 7.374516772941995e-14 * alt**2) 
    
    gav_vec = np.array([[0], [0], [g]])

    return gav_vec

@dataclass
class LooseGnssIns:
    """Test"""
    gps_data: np.array
    imu_data: np.array
    sigma_p: float = 3         # m
    sigma_v: float = 0.2       # m/s
    sigma_ba: float = 0.0005   # g*s^1/2
    tau_s: float = 300         # s
    sigma_wa: float = 0.12     # g
    sigma_bg: float = 0.3      # deg/s^1/2
    tau_g: float = 300         # s
    sigma_wg: float = 0.95     # deg/s
    acc_bias: np.array = np.array([[0.25], [0.077], [-0.12]])
    gyro_bias: np.array = np.array([[2.4], [-1.3], [5.6]]) * 1e-4
    
    def vec_to_skew(self) -> np.array:

        skew_mat = 1

        return skew_mat
    
    def transpose_vec(self, vec: np.array) -> np.array:

        return vec.reshape((-1, 1))
    
    def update_state_ins(self, time: int) -> np.array:

        return 1

    def initial_state(self) -> np.array:

        position = self.transpose_vec(self.gps_data[0][1:4])
        velocity = self.transpose_vec(self.gps_data[0][4:7])
        euler_angle = np.array([[0], [0], [0]])

        state = np.vstack((position, velocity, euler_angle, self.acc_bias, self.gyro_bias))
        print(state)

        return state

def main():

    gps_data = np.genfromtxt('gps.txt', delimiter=',', dtype=float)
    imu_data = np.genfromtxt('imu.txt', delimiter=',', dtype=float)
    
    data = LooseGnssIns(gps_data, imu_data)

    data.initial_state()

    return 1

if __name__=="__main__":
    main()