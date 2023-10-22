#!/bin/python3

import numpy as np
import matplotlib.pyplot as plt

def main():

    gps_data = np.genfromtxt('gps.txt', delimiter=',', dtype=float)
    imu_data = np.genfromtxt('imu.txt', delimiter=',', dtype=float)

    print(gps_data[0])
    print(gps_data[1])
    print(gps_data[2])

    pass

    return 1

if __name__=="__main__":
    main()