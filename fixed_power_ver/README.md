# O-RAN-RAProb
File này được sử dụng để chạy thuật toán SA, giải bài toán bằng MOSEK và đào tạo mô hình DRL.
Cấu trúc file :
- cvxpy_sol : dùng để tạo bài toán và giải bằng MOSEK
- heuristic_with_DRL : dùng để chạy giải thuật SA và đưa từng bước chạy vào file DRL/Data_DRL
- Qlearn : Thuật toán Qlearning sử dụng để giải bài toán, có chứa Q_table.pkl để chứa bảng Q được lưu sau khi train
- Input : dùng để chứa và nhập input vào
    + input_file.csv : File này được đưa input vào để chạy
    + inputphu.csv : File này được dùng để lưu trữ các input đã dùng nhằm tiện cho theo dõi
- Output : chứa output
    + figures : hình thể hiện hội tụ của nghiệm sử dụng SA, được phân biệt bằng id sau tên ứng với stt của dòng input trong file inputphu.csv (hiện đang có một số input bị mất do người thực hiện quên)
    + output.csv : file chứa output chạy (hiện đang chỉ có kết quả của SA, cvxpy sẽ được cập nhật sau)
- results : chứa output để lưu thông tin phân bổ PRB từ từng RU tới các UE, hiện đang lưu trong file .csv


