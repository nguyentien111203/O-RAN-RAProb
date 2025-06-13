import pickle
import time
from Qlearn.env import Environment
from Qlearn.MultiQ import MultiAgentQLearning
from Input import input 
import csv
import os


# Sử dụng để load q_table từ file
def load_q_table(path):
    with open(path, 'rb') as f:
        q_table = pickle.load(f)
    return q_table


"""
    Hàm này được sử dụng để đào tạo mô hình Qlearning cho ra bảng Q để tái sử dụng
    Input :
        env : Môi trường
        alpha, gamma : các tham số học của Qlearning
    Hàm này sẽ train và lưu bảng Q vào một file .pkl
"""
def train(envir : Environment, alpha : float, gamma : float):

    Q_table = {(uf, ut): 0 for uf in range(envir.numuser) for ut in range(envir.numuser) if uf != ut}
    Q_table[(-1,-1)] = 0

    q_learning = MultiAgentQLearning(envir, envir.numuser, envir.numRU, Q_table, alpha, gamma)

    # Huấn luyện
    q_learning.train(max_episodes=2000)

    q_learning.draw_figure(window=10)

    return q_learning.moving_avgs


"""
    Hàm này được sử dụng để phân bổ tài nguyên theo môi trường hiện tại dựa trên Qtable đã có sẵn
    Input :
        env : môi trường hiện tại
        epsilon : xác suất lựa chọn hành động ngẫu nhiên
        Qtable : bảng Q dùng chung
        alpha, gamma : các tham số đang được sử dụng
    Hàm này sẽ chọn các hành động với giá trị Q tốt nhất và trả về ma trận phân bổ
"""
def allocate(env : Environment, epsilon : float, Qtable : dict, episode : int, alpha : float, gamma : float):
    q_learning = MultiAgentQLearning(env, env.numuser, env.numRU, Qtable, alpha, gamma)

    start = time.time()
    # Chọn hành động với giá trị Q tốt nhất
    allocation = q_learning.run_inference(env, Qtable, epsilon, episode, steps_per_ru=3) 
    end = time.time()

    sol_time = end - start

    return allocation, sol_time


"""
    Ghi allocation_matrix vào file CSV, bao gồm số RU và số UE.

    Parameters:
        allocation_matrix (list of list): Ma trận phân phối PRB [numRU][numUE]
        filename (str): Tên file CSV
        folder (str): Thư mục lưu file
    """
def save_allocation_to_csv(allocation_matrix, filename="allocation_result.csv", folder="results"):
    
    os.makedirs(folder, exist_ok=True)
    filepath = os.path.join(folder, filename)

    num_rus = len(allocation_matrix)
    num_ues = len(allocation_matrix[0]) if num_rus > 0 else 0

    with open(filepath, mode='w', newline='') as file:
        writer = csv.writer(file)

        # Mô tả
        writer.writerow([f"Number of RUs: {num_rus}"])
        writer.writerow([f"Number of UEs: {num_ues}"])
        writer.writerow([])  # Dòng trống

        # Header cột: UE0, UE1, ...
        header = [f"UE{i}" for i in range(num_ues)]
        writer.writerow(["RU\\UE"] + header)

        # Ghi dữ liệu
        for ru_idx, row in enumerate(allocation_matrix):
            writer.writerow([f"RU{ru_idx}"] + list(row))

    print(f"Allocation matrix with RU/UE info saved to: {filepath}")













