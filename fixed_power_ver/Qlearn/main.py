import numpy as np
from env import Environment
from MultiQ import MultiAgentQLearning

def main():
    # Thông số ví dụ
    numuser = 3
    numRU = 2
    B = [52, 52]    # Số RB ở mỗi RU
    PeachRB = [10, 10]  # đơn vị : mW
    
    RminK = np.array([0.1, 0.1, 0.1])  # Mbps
    BandW = 0.18  # 180 kHz
    N0 = 3.98e-12  #mW/MHz

    # Muốn tính toán từ ma trận kênh truyền tới ma trận H cần (ví dụ kênh truyền H[i][k] thành |H[i][k]|):
    # HeachRB = numpy.abs(H)

    # Hàm lấy input ở đây
    # Vi du munuser, numRU, B, PeachRB, HeachRU, RminK, BandW, N0 = takeinput() chẳng hạn

    # Khởi tạo môi trường và thuật toán
    # Tôi đã thêm delta để ông có thể tùy chỉnh mức tăng giảm
    env = Environment(numuser, numRU, B, PeachRB, HeachRU, RminK, BandW, N0, delta = 0.05)
    q_learning = MultiAgentQLearning(env, numuser, numRU)

    # Huấn luyện
    q_learning.train(max_episodes=100)

    # Kết quả
    allocation = q_learning.get_allocation()
    print("Final Allocation_matrix:")
    print(allocation)
    env.compute_throughput()
    print("Final Throughput (R_k):", env.R_k, "Mbps")
    print("RminK:", RminK, "Mbps")

# Ví dụ sử dụng
if __name__ == "__main__":
    main()



# Thông số ví dụ
numuser = 3 # Số UEs cần được phục vụ (cần lấy)
numRU = 1 # Số RU hiện có (cần lấy)
B = [52]  #Số RB ở mỗi RU (cần lấy)
P = [10]  # Công suất mỗi PRB của từng RU (cần lấy)
RminK = np.array([0.1, 0.1, 0.1])  # Yêu cầu của người dùng về data rate (đơn vị Mbps) (cần lấy)
H = [] # Vecto kênh truyền, tôi đang tự để là nó chỉ phụ thuộc vào khoảng cách. (có thể tự cho, tôi có hàm để cho)
BandW = 0.18  # Băng thông mỗi PRB chiếm (MHz) (cần lấy)
Thrmin = 1 # Giá trị scale theo tổng throughput, tránh để nó ảnh hưởng quá lớn tới hàm mục tiêu (Đang để là Mbps, tự cho)
N0 = 3.98e-12  # Mật độ công suất nhiễu (mW/MHz) (có thể tính vì đây là nhiễu trắng)
