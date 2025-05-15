import cvxpy_sol.ILPsolver as ILPsolver
import Input.input as input
from Greedyalgo.greedy import GreedyAllocation
import csv
import createtest
import ast
import common
import time
from Qlearn import env, MultiQ

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
            # Khởi tạo môi trường và thuật toán
            envir = env.Environment(numuser, numRU, B, P, H, RminK, BandW, N0, delta = 1)
            q_learning = MultiQ.MultiAgentQLearning(envir, numuser, numRU)
            
            # Huấn luyện
            
            q_learning.train(max_episodes=20)

            end = time.time()  

            # Kết quả
            allocation = q_learning.get_allocation()
            #print("Final Allocation_matrix:")
            print(allocation)
            envir.compute_throughput()
            print("Final Throughput (R_k):", envir.R_k, "Mbps")
            print("RminK:", RminK, "Mbps")

            throughput = envir.R_k
            
            num_served = sum(1 for k in K if throughput[k] >= RminK[k])

            obj_Q = (1-common.tunning) * (sum(throughput)/Thrmin) + (common.tunning) * num_served 

            greedyProb = GreedyAllocation(numuser, numRU, H, B, P, RminK, Thrmin, BandW, N0)

            greedyInfo = greedyProb.evaluate_demand_ratio()

            createtest.write_data_test("./Output/output.csv", 0, numuser, numRU, B, 
                                time_ILP=prob.time, 
                                throughput_ILP=prob.throughput,
                                numuser_ILP=prob.num_user_serve, 
                                check_ILP=prob.check, 
                                objective_value=prob.objvalue,
                                numuser_Q = num_served, 
                                throughput_Q = sum(throughput[k] for k in K),
                                time_Q = end - start,
                                obj_Q=obj_Q,
                                numuser_greedy=greedyInfo.get('num_served_users'),
                                throughput_greedy=greedyInfo.get('throughput'),
                                time_greedy=greedyInfo.get('time'),
                                obj_greedy=greedyInfo.get('objective')
                                )

    
        
main()
    