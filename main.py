#!/bin/python3

from dataclasses import dataclass
import numpy as np
import matplotlib.pyplot as plt
import navpy as nav

@dataclass
class LooseGnssIns:
    """Test"""
    gps_data: np.array
    imu_data: np.array
    time_hist: np.array = np.zeros((1, 16))
    P: np.array = np.zeros((15, 15))
    DMC: np.array = np.zeros((3, 3))
    sigma_p: float = 3         # m
    sigma_v: float = 0.2       # m/s
    sigma_ba: float = 0.0005   # g*s^1/2
    sigma_bg: float = 0.3      # deg/s^1/2
    tau_a: float = 300         # s
    sigma_wa: float = 0.12     # g
    tau_g: float = 300         # s
    sigma_wg: float = 0.95     # deg/s
    sigma_ua: float = 2 * sigma_ba**2 / tau_a
    sigma_ug: float = 2 * sigma_bg**2 / tau_g
    acc_bias: np.array = np.array([[0.25], [0.077], [-0.12]])
    gyro_bias: np.array = np.array([[2.4], [-1.3], [5.6]]) * 1e-4
    
    def vec_to_skew(self, vec: np.array) -> np.array:

        if vec.ndim == 1:
            skew_mat = np.array([[0, -vec[2], vec[1]], \
                             [vec[2], 0, -vec[0]], \
                             [-vec[1], vec[0], 0]])
        else: 
            skew_mat = np.array([[0, -vec[2][0], vec[1][0]], \
                                [vec[2][0], 0, -vec[0][0]], \
                                [-vec[1][0], vec[0][0], 0]])

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
    
        g = g0 * (1 - (3.157042870579883e-7 - 2.10268965044023439e-9 * np.sin(lat)**2)\
              * alt + 7.374516772941995e-14 * alt**2) 
    
        gav_vec = np.array([[0], [0], [g]])

        return gav_vec
    
    def DCM_calc(self, angles: np.array) -> np.array:

        phi = angles[0]
        theta = angles[1]
        psi = angles[2]
        DCM = np.array([[np.cos(theta) * np.cos(psi), \
                         np.sin(phi) * np.sin(theta) * np.cos(psi) - np.cos(phi) * np.sin(psi), \
                         np.cos(phi) * np.sin(theta) * np.cos(psi) + np.sin(phi) * np.sin(psi)], \
                         [np.cos(theta) * np.sin(psi), \
                         np.sin(phi) * np.sin(theta) * np.sin(psi) - np.cos(phi) * np.cos(psi), \
                         np.cos(phi) * np.sin(theta) * np.sin(psi) + np.sin(phi) * np.cos(psi)], \
                         [-np.sin(theta), \
                         np.sin(phi) * np.cos(theta), \
                         np.cos(phi) * np.cos(theta)]])

        return DCM
    
    def initial_state(self) -> np.array:

        position = self.transpose_vec(self.gps_data[0][1:4])
        velocity = self.transpose_vec(self.gps_data[0][4:7])
        euler_angle = np.array([[0], [0], [0]])

        state = np.vstack((position, velocity, euler_angle, self.acc_bias, self.gyro_bias))

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
        long = position[1]
        h = position[2]
        vNx = velocity[1]
        vNy = velocity[0]
        vNz = velocity[2]
        phi = euler_angles[0]
        theta = euler_angles[1]
        psi = euler_angles[2]

        self.DCM = self.DCM_calc(euler_angles)
        
        # ATTITUDE UPDATE
        Amat = 1 / np.cos(theta) * \
            np.array([[1, np.sin(phi) * np.sin(theta), np.cos(phi) * np.sin(theta)], \
                      [0, np.cos(phi) * np.cos(theta), -np.sin(phi) * np.cos(theta)], \
                      [0, np.sin(phi), np.cos(phi)]])
        
        omega_B_IB = self.imu_data[idx-1][1:4] + gyro_bias
        omega_B_NB = self.transpose_vec(omega_B_IB)

        euler_angles = self.transpose_vec(euler_angles) + time_diff * Amat @ omega_B_NB
        
        # VELOCITY UPDATE
        omega_N_IE = 7.292115e-5 * np.array([[np.cos(lat)], [0], [-np.sin(lat)]])
        omega_N_EN = np.array([[vNy / (self.rad_of_curve('east', lat) + h)], \
                               [-vNx / (self.rad_of_curve('north', lat) + h)], \
                               [-vNy * np.tan(lat) / self.rad_of_curve('east', lat) + h]])
        omega_N_IN = omega_N_IE + omega_N_EN

        g_N = self.gav_vector(lat, h)

        f_N = self.imu_data[idx-1][4:7] + acc_bias
        f_B = self.transpose_vec(f_N)

        v_N_dot = self.DCM @ f_B + g_N
                # - self.vec_to_skew(2 * omega_N_IE + omega_N_EN) \
                # @ self.transpose_vec(velocity)

        velocity = self.transpose_vec(velocity) + time_diff * v_N_dot

        # POSITION UPDATE
        p_E_dot = np.array([[1 / (self.rad_of_curve('north', lat) + h), 0, 0], \
                            [0, 1 / ((self.rad_of_curve('east', lat) + h) * np.cos(lat)), 0], \
                            [0, 0, -1]])

        position = self.transpose_vec(position) + time_diff * p_E_dot @ velocity

        state = np.hstack((self.transpose_vec(position), \
                           self.transpose_vec(velocity), \
                           self.transpose_vec(euler_angles), \
                           self.transpose_vec(self.acc_bias), \
                           self.transpose_vec(self.gyro_bias)))
        
        # COVARIANCE UPDATE
        A_mat = np.block([[-self.vec_to_skew(omega_N_EN), np.eye(3), np.zeros((3, 3)), \
                           np.zeros((3, 3)), np.zeros((3, 3))], \
                          [np.linalg.norm(g_N) / a * np.diagflat([-1, -1, 2]), \
                           -self.vec_to_skew(2 * omega_N_IE + omega_N_EN), \
                           self.vec_to_skew(self.DCM @ f_B), self.DCM, np.zeros((3, 3))], \
                          [np.zeros((3, 3)), np.zeros((3, 3)), -self.vec_to_skew(omega_N_IN), \
                           np.zeros((3, 3)), -self.DCM], \
                          [np.zeros((3, 3)), np.zeros((3, 3)), np.zeros((3, 3)), \
                           -1 / self.tau_g * np.eye(3), np.zeros((3, 3))], \
                          [np.zeros((3, 3)), np.zeros((3, 3)), np.zeros((3, 3)), np.zeros((3, 3)), \
                           -1 / self.tau_a * np.eye(3)]])
        
        L_mat = np.block([[np.zeros((3, 3)), np.zeros((3, 3)), np.zeros((3, 3)), np.zeros((3, 3))], \
                          [self.DCM, np.zeros((3, 3)), np.zeros((3, 3)), np.zeros((3, 3))], \
                          [np.zeros((3, 3)), -self.DCM, np.zeros((3, 3)), np.zeros((3, 3))], \
                          [np.zeros((3, 3)), np.zeros((3, 3)), np.eye(3), np.zeros((3, 3))], \
                          [np.zeros((3, 3)), np.zeros((3, 3)), np.zeros((3, 3)), np.eye(3)]])
        
        S_PSD = np.block([[self.sigma_wa**2 * np.eye(3), np.zeros((3, 3)), \
                           np.zeros((3, 3)), np.zeros((3, 3))], \
                          [np.zeros((3, 3)), self.sigma_wg**2 * np.eye(3), \
                           np.zeros((3, 3)), np.zeros((3, 3))], \
                          [np.zeros((3, 3)), np.zeros((3, 3)), \
                           self.sigma_ua**2 * np.eye(3), np.zeros((3, 3))], \
                          [np.zeros((3, 3)), np.zeros((3, 3)), \
                           np.zeros((3, 3)), self.sigma_ug**2 * np.eye(3)]])

        F_mat = np.exp(A_mat * time_diff)

        Q_mat = (np.eye(15) + time_diff * A_mat) \
        * (time_diff * L_mat @ S_PSD @ np.transpose(L_mat))

        self.P = F_mat @ self.P @ np.transpose(F_mat) + Q_mat

        return state
    
    def update_state_ES_EKF(self, idx: int, cur_state: np.array) -> np.array:
        
        ref_location_lla = cur_state[0:3]
        gps_location_lla = self.gps_data[idx][1:4]

        gps_location_ned = nav.lla2ned(gps_location_lla[0], \
                                   gps_location_lla[1], \
                                   gps_location_lla[2], \
                                   ref_location_lla[0], \
                                   ref_location_lla[1], \
                                   ref_location_lla[2], 
                                   latlon_unit='rad')

        gps_y_vec = np.block([[self.transpose_vec(gps_location_ned)], \
                              [self.transpose_vec(self.gps_data[idx][4:7])]])
        ins_y_vec = np.block([[np.zeros((3,1))], [self.transpose_vec(cur_state[4:7])]])

        del_y = gps_y_vec - ins_y_vec

        H_k = np.block([np.eye(6), np.zeros((6, 9))])

        S_k = H_k @ self.P @ np.transpose(H_k)

        K_k = self.P @ np.transpose(H_k) @ np.linalg.inv(S_k)

        error_state = K_k @ del_y

        self.P = self.P - K_k @ S_k @ np.transpose(K_k)

        position = cur_state[0:2] + \
                      np.array([error_state[0][0] \
                      / (self.rad_of_curve('north', cur_state[0]) + cur_state[2]), \
                      error_state[1][0] \
                      / ((self.rad_of_curve('east', cur_state[0]) + cur_state[2]) \
                      * np.cos(cur_state[0]))])
        
        position = np.append(position, self.gps_data[idx][3])

        velocity = cur_state[3:5] + \
                      np.array([error_state[3][0], error_state[4][0]])

        velocity = np.append(velocity, self.gps_data[idx][6])

        self.DCM = (np.eye(3) - self.vec_to_skew(error_state[6:9])) @ self.DCM

        euler_angles = np.array([np.arctan(self.DCM[2][1] / self.DCM[2][2]), \
                                -np.arcsin(self.DCM[2][0]), \
                                np.arctan(self.DCM[1][0] / self.DCM[0][0])])

        self.acc_bias = self.acc_bias + error_state[9:12]

        self.gyro_bias = self.gyro_bias + error_state[12:15]

        state = np.hstack((self.transpose_vec(position), \
                           self.transpose_vec(velocity), \
                           self.transpose_vec(euler_angles), \
                           self.acc_bias, \
                           self.gyro_bias))

        return state
    
    def compute_time_history(self) -> np.array:

        for idx, (gps, imu) in enumerate(zip(self.gps_data, self.imu_data)):
            if idx == 0:
                cur_state = self.initial_state()
                self.time_hist = np.insert(cur_state, 0, self.gps_data[0][0])
                self.P = 10 * np.eye(15)
            else:
                cur_state = self.update_state_ins(idx)
                if np.array_equal(gps, old_gps) == False:
                    cur_state = self.update_state_ES_EKF(idx, cur_state)
                self.time_hist = np.vstack((self.time_hist, \
                                            np.insert(cur_state, 0, self.gps_data[idx][0])))
            old_gps = gps
            old_imu = imu

        return self.time_hist

def main():

    gps_data = np.genfromtxt('gps.txt', delimiter=',', dtype=float)
    imu_data = np.genfromtxt('imu.txt', delimiter=',', dtype=float)
    
    data = LooseGnssIns(gps_data, imu_data)

    time_hist = data.compute_time_history()

    plt.plot(time_hist[:, 1], time_hist[:, 2])

    plt.show()

    return 1

if __name__=="__main__":
    main()