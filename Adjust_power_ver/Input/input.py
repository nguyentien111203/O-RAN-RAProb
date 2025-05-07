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
   
    # B: mảng 2 chiều (RU, RB): mỗi hàng là chỉ số RB của RU đó (pad bằng -1 nếu không đủ)
    maxRB = max(numRBeRU)
    B = []  
    for i in range(numRU):
        B.append(np.arange(numRBeRU[i]))

    # H: mảng 3 chiều (RU, RB, User), pad bằng 0 ở vị trí RB không hợp lệ
    H = np.zeros((numRU, maxRB, numuser))
    for i in range(numRU):
        for b in range(numRBeRU[i]):
            H[i, b] = np.random.uniform(0.1, 1.0, size=numuser)
 
    return K, I, B, H

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



