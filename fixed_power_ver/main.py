import cvxpy_sol.ILPsolver as ILPsolver
import Input.input as input
from Greedyalgo.greedy import greedySolve
import csv
import createtest as createtest
import ast
import common as common
import time
from Qlearn import env, utils

"""
    Hàm này để áp dụng vào xApp (hiện đang để cố định với mẫu hiện tại, sẽ có chỉnh sửa sau)
    Input : 
        numuser, numRU, B, P, H, RminK, Thrmin, BandW, N0 : Đầu vào của bài toán
        delta : số PRB mà từng hành động phân bổ từ người này sang người khác
        episode : số vòng cập nhật để agent thử và thực hiện hành động
        epsilon : xác suất để lựa chọn hành động bất kỳ
        alpha : learning rate)
        gamma : discount factor
    Output :
        allocation : Ma trận phân bổ
"""
def applyXApp(numuser, numRU, B, P, H, RminK, Thrmin, BandW, N0, 
              delta = 1, episode = 10, epsilon = 0.05, alpha = 0.15, gamma = 0.8):
    envir = env.Environment(numuser, numRU, B, P, H, RminK, Thrmin, BandW, N0, delta=1)
            
    allocation, time_Q = utils.allocate(envir, 0.05, Q_table_path="./Qlearn/Q_table.pkl", 
                                        episode = 10, alpha = 0.15, gamma = 0.8)

    return allocation, time_Q

def main():
    
    #for file in os.listdir("./Input/Input_data"): 
    # Mở CSV input_file với open
    input_file = "./Input/input_file.csv"
    # Đường dẫn tới file mô hình
    model_path = "./ppo_rb_allocation.zip"

    with open(input_file, mode = 'r') as csvfile:
        # Tạo một csv reader
        reader = csv.reader(csvfile)
        next(reader)
        for row in reader:
            numuser, numRU, B, P, RminK, Thrmin, BandW, N0 = row

            numuser = int(numuser)
            numRU = int(numRU)
            B = ast.literal_eval(B)
            P = ast.literal_eval(P)
            RminK = ast.literal_eval(RminK)
            Thrmin = float(Thrmin)
            BandW = float(BandW)
            N0 = float(N0)
            
            # Tạo đầu vào cho bài toán
            K, I, H  = input.createEnvironmentInput(numuser, numRU)

            # Giải bài toán với ILP sử dụng cvxpy với mosek
            time_ILP, throughput_ILP, serve_ILP, check_ILP, objvalue_ILP = ILPsolver.solveILP(K, I, H, B, P, RminK, Thrmin, BandW, N0)
            
            # Giải bài toán với Qlearning với 
            # Khởi tạo môi trường và thuật toán
            envir = env.Environment(numuser, numRU, B, P, H, RminK, Thrmin, BandW, N0, delta=1)
            
            #Train đã thực hiện từ trước với việc lựa chọn tham số phù hợp
            
            #moving_avgs = train.train(envir, alpha = 0.1, gamma = 0.65)
            #reward_alpha.append(moving_avgs)
            
            allocation, time_Q = applyXApp(numuser, numRU, B, P, H, RminK, Thrmin, BandW, N0, 
                                           delta = 1, episode = 10, epsilon = 0.05, alpha = 0.15, gamma = 0.8)
            utils.save_allocation_to_csv(allocation, filename="allocation_UE_RU.csv", folder = "results")
            
            throughput = envir.R_k
            
            num_served = sum(1 for k in K if throughput[k] >= RminK[k])

            obj_Q = (1-common.tunning) * (sum(throughput/Thrmin)) + (common.tunning) * num_served 

            # Giải bằng thuật toán baseline
            served_greedy, thr_greedy, time_greedy, obj_greedy = greedySolve(numuser, numRU, H, B, P, RminK, Thrmin, BandW, N0)
            
            createtest.write_data_test("./Output/output.csv", 0, numuser, numRU, B, 
                                time_ILP=time_ILP, 
                                throughput_ILP=throughput_ILP,
                                numuser_ILP=serve_ILP, 
                                check_ILP=check_ILP, 
                                objective_value=objvalue_ILP,
                                numuser_Q = num_served, 
                                throughput_Q = sum(throughput[k] for k in K),
                                time_Q = time_Q,
                                obj_Q=obj_Q,
                                numuser_greedy=served_greedy,
                                throughput_greedy=thr_greedy,
                                time_greedy=time_greedy,
                                obj_greedy=obj_greedy
                                )
                
            
            
    
        
main()
    