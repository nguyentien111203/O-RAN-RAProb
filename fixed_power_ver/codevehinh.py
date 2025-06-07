import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

# Dữ liệu mẫu từ bảng bạn đã đưa ra
data = {
    "numuser": [15, 20, 25, 30],
    "numuser_ILP": [100, 100, 100, 100],
    "throughput_ILP": [485.157, 440.322, 426.469, 449.840],
    "obj_ILP": [15.470, 20.420, 25.401, 30.420],

    "numuser_Q": [100, 100, 92, 83.33],
    "throughput_Q": [410.006, 379.397, 373.686, 398.665],
    "obj_Q": [15.395, 20.359, 23.350, 25.373],

    "numuser_greedy": [100, 90, 80, 46.66],
    "throughput_greedy": [337.441, 306.147, 302.885, 305.204],
    "obj_greedy": [15.322, 18.288, 20.282, 14.291]
}

# Thiết lập font và dùng LaTeX
plt.rcParams.update({
    "text.usetex": True,             # Dùng LaTeX để hiển thị text
    "font.family": "serif",          # Font serif (giống trong tài liệu khoa học)
    "font.size": 35,                 # Tăng cỡ chữ chung (toàn biểu đồ)
    "legend.fontsize": 35,           # Cỡ chữ cho chú thích
    "axes.labelsize": 35,            # Cỡ chữ cho nhãn trục
    "xtick.labelsize": 35,
    "ytick.labelsize": 35,
})

x = np.arange(len(data["numuser"]))
width = 0.2 # Độ rộng mỗi cột
space = 0.03
df = pd.DataFrame(data)

plt.figure(figsize=(12, 8))
plt.bar(x - width - space, df["numuser_ILP"], width, label="ILP",color='white', edgecolor='green', hatch='.')
plt.bar(x, df["numuser_Q"], width, label="Q-learning",color='white', edgecolor='orange', hatch='//')
plt.bar(x + width + space, df["numuser_greedy"], width, label="Greedy",color='white', edgecolor='blue', hatch='\\')

plt.xlabel("Number of users")
plt.ylabel(r"Acceptance rate (\%)")
plt.xticks(x, df["numuser"])  # gắn nhãn trục x theo số người dùng
plt.ylim(bottom = 0, top = 110)
plt.legend(loc = "lower left")
plt.grid(True, axis='y', linestyle='--', alpha=0.7)
plt.grid(True, axis='x', linestyle='--', alpha=0.7)
plt.tight_layout()
plt.savefig("./Picture/ue_compare.png", format = "png")
plt.show()

plt.figure(figsize=(12, 8))
plt.bar(x - width - space, df["throughput_ILP"], width, label="ILP",color='white', edgecolor='green', hatch='.')
plt.bar(x, df["throughput_Q"], width, label="Q-learning",color='white', edgecolor='orange', hatch='//')
plt.bar(x + width + space, df["throughput_greedy"], width, label="Greedy",color='white', edgecolor='blue', hatch='\\')

plt.xlabel("Number of users")
plt.ylabel("Sum of data rate (Mbps)")
plt.xticks(x, df["numuser"])  # gắn nhãn trục x theo số người dùng
plt.legend(loc = "lower left")
plt.grid(True, axis='y', linestyle='--', alpha=0.7)
plt.grid(True, axis='x', linestyle='--', alpha=0.7)
plt.tight_layout()
plt.savefig("./Picture/thr_compare.png")
plt.show()


# 3. So sánh giá trị hàm mục tiêu
plt.figure(figsize=(12, 8))
plt.bar(x - width - space, df["obj_ILP"], width, label="ILP",color='white', edgecolor='green', hatch='.')
plt.bar(x, df["obj_Q"], width, label="Q-learning",color='white', edgecolor='orange', hatch='//')
plt.bar(x + width + space, df["obj_greedy"], width, label="Greedy",color='white', edgecolor='blue', hatch='\\')

plt.xlabel("Number of users")
plt.ylabel("Objective")
plt.xticks(x, df["numuser"])  # gắn nhãn trục x theo số người dùng
plt.legend(loc = "lower left")
plt.grid(True, axis='y', linestyle='--', alpha=0.7)
plt.grid(True, axis='x', linestyle='--', alpha=0.7)
plt.tight_layout()
plt.savefig("./Picture/obj_compare.png")
plt.show()
