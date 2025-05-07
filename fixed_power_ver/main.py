import cvxpy_sol.ILPsolver as ILPsolver
import Input.input as input
from Greedyalgo.greedy import GreedyAllocation
import csv
import createtest
import ast
import torch
import numpy as np
import os
from stable_baselines3 import PPO  # Sử dụng mô hình đã được train
import os
from PPOtrain import train  # Thay thế bằng môi trường thực tế của bạn
import time
from Qlearn import Qtrain

def main():
    """
    # Bước 1: Load dữ liệu từ file
    X, y, numuser = train.build_dataset_from_csv("./Input/input_file.csv")

    # Bước 2: Huấn luyện mô hình
    model = train.train_model(X, y, numuser)

    # Bước 3: Lưu lại mô hình
    train.save_model(model)
    """
    #for file in os.listdir("./Input/Input_data"): 
    # Mở CSV input_file với open
    input_file = "./input_real.csv"
    # Đường dẫn tới file mô hình
    model_path = "./ppo_rb_allocation.zip"

    with open(input_file, mode = 'r') as csvfile:
        # Tạo một csv reader
        reader = csv.reader(csvfile)
        next(reader)
        for row in reader:
            numuser, numRU, RBeachRU, Pmax, RminK, Thrmin, BandW, N0 = row

            numuser = int(numuser)
            numRU = int(numRU)
            RBeachRU = ast.literal_eval(RBeachRU)
            Pmax = ast.literal_eval(Pmax)
            RminK = ast.literal_eval(RminK)
            Thrmin = float(Thrmin)
            BandW = float(BandW)
            N0 = float(N0)
            
            # Tạo đầu vào cho bài toán
            K, I, B, H, P  = input.createEnvironmentInput(numuser, numRU, RBeachRU, Pmax)

            """
            K, I, B, H, RminK, Pmax, Thrmin, BW, N0 = input.input_from_npz("./Input/Input_data/" + file)
            """
            # Giải bài toán với CVXPY
            
            prob = ILPsolver.AllocationProblemILP(K = K, I = I, H = H, B = B, P = P,
                                            RminK = RminK, Thrmin = Thrmin, BandW = BandW, N0 = N0)

            # RminK : Mbps, BW : MHz, N0 : mW/MHz, Pmax : mW

            prob.solve()

            # Đưa vào tensor
            start = time.time()
            ## === Trích đặc trưng ===
            X, rb_map = Qtrain.extract_pointwise_features(K, I, B, H, P, RminK, Thrmin, BandW, N0)

            # === Sinh nhãn giả định (Q target) để huấn luyện thử ===
            y_dummy = torch.rand(len(X), 1)

            # === Huấn luyện ===
            model = Qtrain.train_model_qscore(X, y_scores=y_dummy, epochs=10)

            # === Dự đoán phân bổ ===
            allocation = Qtrain.predict_allocation_q_scoring(model, K, I, B, H, P, RminK, Thrmin, BandW, N0)
            
            end = time.time()  

            throughput, num_served = Qtrain.compute_throughput_and_served_users(K = K, I=I, B=B, H=H, P=P, RminK=RminK, 
                                                            BandW=BandW, N0 = N0, allocation=allocation)

            
            createtest.write_data_test("./Output/output.csv", 0, numuser, numRU, RBeachRU, 
                                prob.time, 
                                prob.throughput,
                                prob.num_user_serve, 
                                prob.check, 
                                numuser_SA = num_served, 
                                throughput_SA = sum(throughput[k] for k in K),
                                time_SA = end - start
                                )

    
        
main()
    