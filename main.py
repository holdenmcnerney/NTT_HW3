#!/bin/python3

import numpy as np
import matplotlib.pyplot as plt
import navpy as nav
import scipy.linalg as scila
from dataclasses import dataclass

# Matplotlib global params
plt.rcParams['axes.grid'] = True

@dataclass
class LooseGnssIns:
    """Test"""
    gps_data: np.array
    imu_data: np.array
    time_hist: np.array = np.zeros((1, 16))
    P: np.array = np.zeros((15, 15))
    DCM: np.array = np.zeros((3, 3))
    sigma_p: float = 3                                              # m
    sigma_v: float = 0.2                                            # m/s
    sigma_ba: float = 0.0005                                        # g s^1/2
    sigma_bg: float = 0.3 * np.pi / 180                             # rad/s^1/2
    tau_a: float = 300                                              # s
    sigma_wa: float = 0.12                                          # g s^1/2
    tau_g: float = 300                                              # s
    sigma_wg: float = 0.95 * np.pi / 180                            # rad/s
    acc_bias: np.array = np.array([[0.25], [0.077], [-0.12]])       # m/s*2
    gyro_bias: np.array = np.array([[2.4], [-1.3], [5.6]]) * 1e-4   # rad/s
    
    def vec_to_skew(self, vec: np.array) -> np.array:

        if vec.ndim == 1:
            skew_mat = np.array([[0, vec[2], -vec[1]], \
                             [-vec[2], 0, vec[0]], \
                             [vec[1], -vec[0], 0]])
        else: 
            skew_mat = np.array([[0, vec[2][0], -vec[1][0]], \
                                [-vec[2][0], 0, vec[0][0]], \
                                [vec[1][0], -vec[0][0], 0]])

        return skew_mat
    
    def transpose_vec(self, vec: np.array) -> np.array:
        
        if vec.ndim == 1:
            vec = np.array([[vec[0]], [vec[1]], [vec[2]]])
        else:
            vec = np.array([vec[0][0], vec[1][0], vec[2][0]])

        return vec
    
    def rad_of_curve(self, dir: str, lat: float) -> float:

        a = 6378137
        f = 1/298.257223563
        e = np.sqrt(f * (2 - f))

        if dir == 'north':
            rad = (a * (1 - e**2)) / (1 - e**2 * np.sin(lat)**2)**1.5
        elif dir == 'east':
            rad = a / np.sqrt(1 - e**2 * np.sin(lat)**2)

        return rad
    
    def gav_vector(self, lat: float, alt:float):

        g0 = 9.7803253359 * (1 + 0.001931853 * np.sin(lat)**2) \
            / np.sqrt(1 - 0.00669438 * np.sin(lat)**2)
        g = g0 * (1 - (3.157042870579883e-7 - 2.10268965044023439e-9 \
               * np.sin(lat)**2)\
               * alt + 7.374516772941995e-14 * alt**2) 
        gav_vec = np.array([[0], [0], [g]])

        return gav_vec
    
    def DCM_calc(self, angles: np.array) -> np.array:

        phi = angles[0]
        theta = angles[1]
        psi = angles[2]
        DCM = np.array([[np.cos(theta) * np.cos(psi), \
                         np.sin(phi) * np.sin(theta) * np.cos(psi) \
                         - np.cos(phi) * np.sin(psi), \
                         np.cos(phi) * np.sin(theta) * np.cos(psi) \
                         + np.sin(phi) * np.sin(psi)], \
                         [np.cos(theta) * np.sin(psi), \
                         np.sin(phi) * np.sin(theta) * np.sin(psi) \
                         + np.cos(phi) * np.cos(psi), \
                         np.cos(phi) * np.sin(theta) * np.sin(psi) \
                         - np.sin(phi) * np.cos(psi)], \
                         [-np.sin(theta), \
                         np.sin(phi) * np.cos(theta), \
                         np.cos(phi) * np.cos(theta)]])

        return DCM
    
    def initial_state(self) -> np.array:

        position = self.transpose_vec(self.gps_data[0][1:4])
        velocity = self.transpose_vec(self.gps_data[0][4:7])
        euler_angle = np.array([[0], [0], [0]])
        state = np.vstack((position, \
                           velocity, \
                           euler_angle, \
                           self.acc_bias, \
                           self.gyro_bias))

        return state
    
    def update_state_ins(self, idx: int) -> np.array:

        a = 6378137
        time_diff = self.gps_data[idx][0] - self.gps_data[idx - 1][0]

        if self.time_hist.ndim == 1:
            position = self.time_hist[1:4]
            velocity = self.time_hist[4:7]
            euler_angles = self.time_hist[7:10]
            acc_bias = self.time_hist[10:13]
            gyro_bias = self.time_hist[13:16]
        else:
            position = self.time_hist[idx - 1][1:4]
            velocity = self.time_hist[idx - 1][4:7]
            euler_angles = self.time_hist[idx - 1][7:10]
            acc_bias = self.time_hist[idx - 1][10:13]
            gyro_bias = self.time_hist[idx - 1][13:16]

        lat = position[0]
        h = position[2]
        vNx = velocity[0]
        vNy = velocity[1]
        phi = euler_angles[0]
        theta = euler_angles[1]

        # INITIALIZE DCM
        self.DCM = self.DCM_calc(euler_angles)

        # ATTITUDE UPDATE
        A_euler = 1 / np.cos(theta) * \
                    np.array([[1, np.sin(phi) * np.sin(theta), \
                               np.cos(phi) * np.sin(theta)], \
                              [0, np.cos(phi) * np.cos(theta), \
                               -np.sin(phi) * np.cos(theta)], \
                              [0, np.sin(phi), np.cos(phi)]])
        omega_N_IE = 7.292115e-5 * np.array([[np.cos(lat)], \
                                             [0], \
                                             [-np.sin(lat)]])
        omega_N_EN = np.array([[vNy / (self.rad_of_curve('east', lat) + h)], \
                               [-vNx / (self.rad_of_curve('north', lat) + h)], \
                               [-vNy * np.tan(lat) \
                                / (self.rad_of_curve('east', lat) + h)]])
        omega_N_IN = omega_N_IE + omega_N_EN
        omega_B_IB = self.imu_data[idx][1:4] + gyro_bias
        omega_B_NB = self.transpose_vec(omega_B_IB) \
                    - np.linalg.inv(self.DCM) @ self.transpose_vec(omega_N_IN)
        euler_angles = self.transpose_vec(euler_angles) \
                       + time_diff * A_euler @ omega_B_NB
        
        # VELOCITY UPDATE
        f_B_t = self.imu_data[idx][4:7] + acc_bias
        f_B = self.transpose_vec(f_B_t)
        g_N = self.gav_vector(lat, h)
        v_N_dot = self.DCM @ f_B + g_N \
                - self.vec_to_skew(2 * omega_N_IE + omega_N_EN) \
                @ self.transpose_vec(velocity)
        velocity = self.transpose_vec(velocity) + time_diff * v_N_dot

        # POSITION UPDATE
        p_E_dot = np.array([[1 / (self.rad_of_curve('north', lat) + h), 0, 0], \
                            [0, 1 / ((self.rad_of_curve('east', lat) \
                                      + h) * np.cos(lat)), 0], \
                            [0, 0, -1]])
        position = self.transpose_vec(position) + time_diff * p_E_dot @ velocity

        # UPDATING VALUES FOR COVARIANCE UPDATE
        lat = position[0][0]
        h = position[2][0]
        vNx = velocity[0][0]
        vNy = velocity[1][0]
        phi = euler_angles[0][0]
        theta = euler_angles[1][0]
        g_N = self.gav_vector(lat, h)
        omega_N_IE = 7.292115e-5 * np.array([[np.cos(lat)], \
                                             [0], \
                                             [-np.sin(lat)]])
        omega_N_EN = np.array([[vNy / (self.rad_of_curve('east', lat) + h)], \
                               [-vNx / (self.rad_of_curve('north', lat) + h)], \
                               [-vNy * np.tan(lat) \
                                / (self.rad_of_curve('east', lat) + h)]])
        omega_N_IN = omega_N_IE + omega_N_EN
        self.DCM = self.DCM_calc(self.transpose_vec(euler_angles))
        
        # COVARIANCE UPDATE
        A_mat = np.block([[-self.vec_to_skew(omega_N_EN), np.eye(3), \
                           np.zeros((3, 3)), np.zeros((3, 3)), \
                           np.zeros((3, 3))], \
                          [np.linalg.norm(g_N) / a * np.diagflat([-1, -1, 2]), \
                           -self.vec_to_skew(2 * omega_N_IE + omega_N_EN), \
                           self.vec_to_skew(self.DCM @ f_B), self.DCM, \
                           np.zeros((3, 3))], \
                          [np.zeros((3, 3)), np.zeros((3, 3)), \
                           -self.vec_to_skew(omega_N_IN), \
                           np.zeros((3, 3)), -self.DCM], \
                          [np.zeros((3, 3)), np.zeros((3, 3)), \
                           np.zeros((3, 3)), -1 / self.tau_a * np.eye(3), \
                           np.zeros((3, 3))], \
                          [np.zeros((3, 3)), np.zeros((3, 3)), \
                           np.zeros((3, 3)), np.zeros((3, 3)), \
                           -1 / self.tau_g * np.eye(3)]])
        L_mat = np.block([[np.zeros((3, 3)), np.zeros((3, 3)), \
                           np.zeros((3, 3)), np.zeros((3, 3))], \
                          [self.DCM, np.zeros((3, 3)), np.zeros((3, 3)), \
                           np.zeros((3, 3))], \
                          [np.zeros((3, 3)), -self.DCM, np.zeros((3, 3)), \
                           np.zeros((3, 3))], \
                          [np.zeros((3, 3)), np.zeros((3, 3)), np.eye(3), \
                           np.zeros((3, 3))], \
                          [np.zeros((3, 3)), np.zeros((3, 3)), \
                           np.zeros((3, 3)), np.eye(3)]])
        self.sigma_ba = 0.0005 * np.linalg.norm(g_N)
        self.sigma_wa = 0.12 * np.linalg.norm(g_N)
        sigma_ua_sq = 2 * self.sigma_ba**2 / self.tau_a
        sigma_ug_sq = 2 * self.sigma_bg**2 / self.tau_g
        S_PSD = np.block([[self.sigma_wa**2 * np.eye(3), np.zeros((3, 3)), \
                           np.zeros((3, 3)), np.zeros((3, 3))], \
                          [np.zeros((3, 3)), self.sigma_wg**2 * np.eye(3), \
                           np.zeros((3, 3)), np.zeros((3, 3))], \
                          [np.zeros((3, 3)), np.zeros((3, 3)), \
                           sigma_ua_sq * np.eye(3), np.zeros((3, 3))], \
                          [np.zeros((3, 3)), np.zeros((3, 3)), \
                           np.zeros((3, 3)), sigma_ug_sq * np.eye(3)]])
        F_mat = scila.expm(A_mat * time_diff)
        E22 = scila.expm(time_diff * np.transpose(A_mat))
        E12 = scila.expm(time_diff * L_mat @ S_PSD @ np.transpose(L_mat))
        Q_mat = np.transpose(E22) @ E12
        self.P = F_mat @ self.P @ np.transpose(F_mat) + Q_mat

        # STATE UPDATE
        state = np.hstack((self.transpose_vec(position), \
                           self.transpose_vec(velocity), \
                           self.transpose_vec(euler_angles), \
                           self.transpose_vec(self.acc_bias), \
                           self.transpose_vec(self.gyro_bias)))

        return state
    
    def update_state_ES_EKF(self, idx: int, cur_state: np.array) -> np.array:
        
        # ES_EKF TIME UPDATE
        ref_location_lla = cur_state[0:3]
        gps_location_lla = self.gps_data[idx][1:4]
        gps_location_ned = nav.lla2ned(gps_location_lla[0], \
                                   gps_location_lla[1], \
                                   gps_location_lla[2], \
                                   ref_location_lla[0], \
                                   ref_location_lla[1], \
                                   ref_location_lla[2], 
                                   latlon_unit='rad')
        gps_y_vec_ned = np.block([[self.transpose_vec(gps_location_ned)], \
                              [self.transpose_vec(self.gps_data[idx][4:7])]])
        ins_y_vec_ned = np.block([[np.zeros((3,1))], \
                                  [self.transpose_vec(cur_state[3:6])]])
        del_y = gps_y_vec_ned - ins_y_vec_ned
        H_k = np.block([np.eye(6), np.zeros((6, 9))])
        R_k = np.diagflat([self.sigma_p, self.sigma_p, self.sigma_p, \
                           self.sigma_v,self.sigma_v, self.sigma_v])
        S_k = H_k @ self.P @ np.transpose(H_k) + np.eye(6) @ R_k @ np.eye(6)
        K_k = self.P @ np.transpose(H_k) @ np.linalg.inv(S_k)
        error_state = K_k @ del_y
        self.P = self.P - K_k @ S_k @ np.transpose(K_k)

        # ES_EKF MEASUREMENT UPDATE
        error_state_position_ned = error_state[0:3]
        position = cur_state[0:3] + \
                      np.array([error_state_position_ned[0][0] \
                      / (self.rad_of_curve('north', cur_state[0]) \
                         + cur_state[2]), \
                      error_state_position_ned[1][0] \
                      / ((self.rad_of_curve('east', cur_state[0]) \
                          + cur_state[2]) * np.cos(cur_state[0])), \
                      - error_state_position_ned[2][0]])
        position[2] =  self.gps_data[idx][3]
        velocity = cur_state[3:6] + \
                   np.array([error_state[3][0], \
                             error_state[4][0], \
                             error_state[5][0]])
        velocity[2] = self.gps_data[idx][6]
        
        # Something is wrong here when adding error state euler angle skew matrix
        # if logic is incorrect but makes the script run
        curDCMval = self.DCM[2][0]
        self.DCM = (np.eye(3) - self.vec_to_skew(error_state[6:9])) @ self.DCM
        if np.isnan(-np.arcsin(self.DCM[2][0])):
            self.DCM[2][0] = curDCMval
        euler_angles = np.array([np.arctan(self.DCM[2][1] / self.DCM[2][2]), \
                                -np.arcsin(self.DCM[2][0]), \
                                np.arctan(self.DCM[1][0] / self.DCM[0][0])])
    
        self.acc_bias = self.acc_bias + error_state[9:12]
        self.gyro_bias = self.gyro_bias + error_state[12:15]

        state = np.hstack((position, \
                           velocity, \
                           euler_angles, \
                           self.transpose_vec(self.acc_bias), \
                           self.transpose_vec(self.gyro_bias)))

        return state
    
    def compute_time_history(self) -> np.array:

        for idx, gps in enumerate(self.gps_data):
            if idx == 0:
                cur_state = self.initial_state()
                self.P = 10 * np.eye(15)
                self.time_hist = np.insert(cur_state, 0, self.gps_data[0][0])
            else:
                cur_state = self.update_state_ins(idx)
                if np.array_equal(gps[1:], old_gps[1:]) == False:
                    cur_state = self.update_state_ES_EKF(idx, cur_state)
                self.time_hist = np.vstack((self.time_hist, \
                                            np.insert(cur_state, \
                                                      0, \
                                                      self.gps_data[idx][0])))
            old_gps = gps

        return self.time_hist
    
    def plot_all(self):

        ref_loc = self.time_hist[0][1:4]
        loc_hist_ned = nav.lla2ned(self.time_hist[:, 1], \
                                   self.time_hist[:, 2], \
                                   self.time_hist[:, 3], \
                                   ref_loc[0], \
                                   ref_loc[1], \
                                   ref_loc[2], \
                                   latlon_unit='rad')

        # FIGURE 1
        fig1, axs1 = plt.subplots(1, 2)
        axs1[0].plot(loc_hist_ned[:, 1], loc_hist_ned[:, 0])
        axs1[0].set_xlabel('East')
        axs1[0].set_ylabel('North')
        axs1[0].set_title('East vs North')

        axs1[1].plot(self.time_hist[:, 0], - loc_hist_ned[:, 2])
        axs1[1].set_xlabel(r'Time, $s$')
        axs1[1].set_ylabel('Altitude')
        axs1[1].set_title('Altitude vs Time')
        
        # FIGURE 2
        fig2, axs2 = plt.subplots(4, 3)
        axs2[0, 0].plot(self.time_hist[:, 0], self.time_hist[:, 4])
        axs2[0, 0].set_ylabel('North Velocity')
        axs2[0, 0].set_title('North Velocity vs Time')

        axs2[0, 1].plot(self.time_hist[:, 0], self.time_hist[:, 5])
        axs2[0, 1].set_ylabel('East Velocity')
        axs2[0, 1].set_title('East Velocity vs Time')

        axs2[0, 2].plot(self.time_hist[:, 0], self.time_hist[:, 6])
        axs2[0, 2].set_ylabel('Down Velocity')
        axs2[0, 2].set_title('Down Velocity vs Time')

        axs2[1, 0].plot(self.time_hist[:, 0], self.time_hist[:, 7])
        axs2[1, 0].set_ylabel('Roll')
        axs2[1, 0].set_title('Roll vs Time')

        axs2[1, 1].plot(self.time_hist[:, 0], self.time_hist[:, 8])
        axs2[1, 1].set_ylabel('Pitch')
        axs2[1, 1].set_title('Pitch vs Time')

        axs2[1, 2].plot(self.time_hist[:, 0], self.time_hist[:, 9])
        axs2[1, 2].set_ylabel('Yaw')
        axs2[1, 2].set_title('Yaw vs Time')

        axs2[2, 0].plot(self.time_hist[:, 0], self.time_hist[:, 10])
        axs2[2, 0].set_ylabel('Accelerometer X Bias')
        axs2[2, 0].set_title('Accelerometer X Bias vs Time')

        axs2[2, 1].plot(self.time_hist[:, 0], self.time_hist[:, 11])
        axs2[2, 1].set_ylabel('Accelerometer Y Bias')
        axs2[2, 1].set_title('Accelerometer Y Bias vs Time')

        axs2[2, 2].plot(self.time_hist[:, 0], self.time_hist[:, 12])
        axs2[2, 2].set_ylabel('Accelerometer Z Bias')
        axs2[2, 2].set_title('Accelerometer Z Bias vs Time')

        axs2[3, 0].plot(self.time_hist[:, 0], self.time_hist[:, 13])
        axs2[3, 0].set_xlabel(r'Time, $s$')
        axs2[3, 0].set_ylabel('Gyroscope X Bias')
        axs2[3, 0].set_title('Gyroscope X Bias vs Time')

        axs2[3, 1].plot(self.time_hist[:, 0], self.time_hist[:, 14])
        axs2[3, 1].set_xlabel(r'Time, $s$')
        axs2[3, 1].set_ylabel('Gyroscope Y Bias')
        axs2[3, 1].set_title('Gyroscope Y Bias vs Time')

        axs2[3, 2].plot(self.time_hist[:, 0], self.time_hist[:, 15])
        axs2[3, 2].set_xlabel(r'Time, $s$')
        axs2[3, 2].set_ylabel('Gyroscope Z Bias')
        axs2[3, 2].set_title('Gyroscope Z Bias vs Time')
        plt.show()

        return 1

def main():

    gps_data = np.loadtxt('gps.txt', delimiter=',', dtype=float)
    imu_data = np.loadtxt('imu.txt', delimiter=',', dtype=float)
    
    data = LooseGnssIns(gps_data, imu_data)
    data.compute_time_history()
    data.plot_all()

    return 1

if __name__=="__main__":
    main()