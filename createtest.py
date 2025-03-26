import csv

"""
    Lưu thông tin vào file output (csv)
"""
def write_data_test(output_file : str, id : int, numuser : int, numRU : int, RBeachRU : set,
                    time_ILP : float, throughput_ILP : float, numuser_ILP : int, check_ILP : bool,
                    step_SA : int, numuser_SA : int, throughput_SA : float, time_SA : float):
    with open(output_file, 'a') as opf:
        writer = csv.writer(opf)
        # Đánh giá nghiệm tốt theo các tiêu chí
        # Thời gian
        
        time_rate = (time_ILP / time_SA)
        
        # Tỷ lệ người dùng
        serve_rate = (numuser_SA/numuser_ILP)*100
        
        # Chênh lệch throughput
        throughput_rate = ((throughput_SA - throughput_ILP)/throughput_ILP)*100
        
        
        writer.writerow([id, numuser,numRU,RBeachRU,time_ILP,throughput_ILP,numuser_ILP,check_ILP,
                         step_SA,numuser_SA,throughput_SA,time_SA,time_rate,serve_rate,throughput_rate])   # Viết các tiêu chí
        
        
