import numpy as np


"""
    Tạo đầu vào liên quan tới số lượng người dùng, số lượng RU và RB
    Đầu vào :
        numuser : số lượng người dùng
        numRU : số lượng RU
        numRBeRU : tập chứa số lượng RB mỗi RU
"""
def createEnvironmentInput(numuser : int, numRU : int, numRBeRU : set):
    K = [k for k in range(numuser)]
    I = [i for i in range(numRU)]
    B = []
    for i in range(numRU):
        B.append([b for b in range(numRBeRU[i])])
    
    H = []
    for k in range(numuser):
        Rbi = []
        for i in range(numRU):
            # Hiện đang thử với tất cả bằng 1 cho dễ theo dõi
            Rbi.append(np.ones(numRBeRU[i]))
        H.append(Rbi)

    return K, I, B, H

