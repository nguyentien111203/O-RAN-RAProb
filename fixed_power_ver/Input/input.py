import numpy as np


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
    H = np.zeros((numRU, numuser))
    for i in range(numRU):
        for k in range(numuser):
            H[i][k] = np.random.uniform(0.7, 1.14)
 
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



