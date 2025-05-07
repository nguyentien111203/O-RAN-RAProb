import csv

"""
    Lưu thông tin vào file output (csv)
"""
def write_data_test(output_file : str, test_id : int ,numuser : int, numRU : int, RBeachRU : set,
                    time_ILP : float, throughput_ILP : float, numuser_ILP : int, check_ILP : bool,
                    numuser_SA : int, throughput_SA : float, time_SA : float):
    with open(output_file, 'a') as opf:
        writer = csv.writer(opf)
        # Đánh giá nghiệm tốt theo các tiêu chí
        
        # Chênh lệch throughput
        throughput_rate = ((throughput_ILP - throughput_SA)/throughput_ILP)*100
        
        
        writer.writerow([test_id, numuser,numRU,RBeachRU,time_ILP,throughput_ILP,numuser_ILP,check_ILP,
                         numuser_SA,throughput_SA,time_SA,throughput_rate])   # Viết các tiêu chí
        
        
