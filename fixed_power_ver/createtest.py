import csv

"""
    Lưu thông tin vào file output (csv)
"""
def write_data_test(output_file : str, test_id : int ,numuser : int, numRU : int, RBeachRU : set,
                    time_ILP : float, throughput_ILP : float, numuser_ILP : int, check_ILP : bool, objective_value : float,
                    numuser_Q : int, throughput_Q : float, time_Q : float, obj_Q : float,
                    numuser_greedy : int, throughput_greedy : float, time_greedy : float, obj_greedy : float):
    with open(output_file, 'a') as opf:
        writer = csv.writer(opf)
        # Đánh giá nghiệm tốt theo các tiêu chí
        
        writer.writerow([test_id, numuser,numRU,RBeachRU,time_ILP,throughput_ILP,numuser_ILP,check_ILP,objective_value,
                         numuser_Q,throughput_Q,time_Q,obj_Q,
                         numuser_greedy, throughput_greedy, time_greedy, obj_greedy])   # Viết các tiêu chí
        
        
