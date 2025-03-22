import csv

"""
    Lưu thông tin vào file output (csv)
"""
def write_data_test(output_file : str, numuser : int, numRU : int, RBeachRU : set,
                    time_ILP : float, numuser_ILP : int, step_SA : int, numuser_SA : int):
    with open(output_file, 'a') as opf:
        writer = csv.writer(opf)
        writer.writerow([numuser,numRU,RBeachRU,time_ILP,numuser_ILP,step_SA,numuser_SA])   # Viết các tiêu chí
        
        
