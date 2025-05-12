import numpy as np
from env import Environment
from MultiQ import MultiAgentQLearning

def main():
    # Thông số ví dụ
    numuser = 3
    numRU = 2
    B = [10, 10]    # Số RB ở mỗi RU
    PeachRB = [0.1, 0.1]  # đơn vị : mW
    HeachRU = np.random.rand(numRU, numuser)  # Đang để ở ví dụ là kênh ngẫu nhiên
    RminK = np.array([2, 3, 4])  # Mbps
    BandW = 0.18  # 180 kHz
    N0 = 1e-6  #mW/Hz

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