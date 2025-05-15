import numpy as np
import math

import numpy as np

def generate_random_distances(num_ru, num_users, d_min=20, d_max=100):
    """
    Sinh ma trận khoảng cách ngẫu nhiên từ mỗi RU đến mỗi user.
    """
    return np.random.uniform(low=d_min, high=d_max, size=(num_ru, num_users))

def compute_H_from_distance(distance_matrix, PL0=30, pathloss_exp=3.5):
    """
    Tính ma trận H (biên độ kênh truyền) từ ma trận khoảng cách.
    """
    PL_dB = PL0 + 10 * pathloss_exp * np.log10(np.maximum(distance_matrix, 1))  # tránh log(0)
    H = 10 ** (-PL_dB / 20)
    return H



"""
    Tạo đầu vào liên quan tới số lượng người dùng, số lượng RU và RB
    Đầu vào :
        numuser : số lượng người dùng
        numRU : số lượng RU
        numRBeRU : tập chứa số lượng RB mỗi RU
"""
def createEnvironmentInput(numuser : int, numRU : int):
    K = np.array([k for k in range(numuser)])  # Danh sách user
    I = np.array([i for i in range(numRU)])  # Danh sách RU

    # H: mảng 2 chiều (RU, User), pad bằng 0 ở vị trí RB không hợp lệ
    #distance = generate_random_distances(numRU, numuser, d_min = 20, d_max = 100)
    #H = compute_H_from_distance(distance_matrix=distance)
    
    H = np.zeros((numRU, numuser))
    for i in range(numRU):
        for k in range(numuser):
            H[i][k] = 0.25**2 + 0.85**2

    return K, I, H




def input_from_npz(file : str):

    data = np.load(file = file, allow_pickle = True)

    # Truy xuất đầu vào theo tên
    K = data["K"]
    I = data["I"]
    B = data["B"]
    H = data["H"]
    RminK = data["RminK"]
    Pmax = data["Pmax"]
    Thrmin = data["Thrmin"]
    BW = data["BW"]
    N0 = data["N0"]

    return K, I, B, H, RminK, Pmax, Thrmin, BW, N0



