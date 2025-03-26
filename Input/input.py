import numpy as np


"""
    Tạo đầu vào liên quan tới số lượng người dùng, số lượng RU và RB
    Đầu vào :
        numuser : số lượng người dùng
        numRU : số lượng RU
        numRBeRU : tập chứa số lượng RB mỗi RU
"""
def createEnvironmentInput(numuser : int, numRU : int, numRBeRU : list):
    K = np.array([k for k in range(numuser)])  # Danh sách user
    I = np.array([i for i in range(numRU)])  # Danh sách RU
    
    # Tạo danh sách B dưới dạng dictionary để giữ số RB khác nhau cho mỗi RU
    B = {i: np.array([b for b in range(numRBeRU[i])]) for i in range(numRU)}

    H = {k: {i: np.random.uniform(0.1, 10, len(B[i])) for i in I} for k in K}
 
    return K, I, B, H

def input_from_npz(file : str):

    data = np.load(file = file)

    # Truy xuất đầu vào theo tên
    K = data["K"]
    I = data["I"]
    B = data["B"]
    H = data["H"]
    RminK = data["RminK"]
    Pmax = data["Pmax"]
    Tmin = data["Tmin"]

    return K, I, B, H, RminK, Pmax, Tmin



