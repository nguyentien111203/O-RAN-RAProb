import numpy as np
import random
import time
import common

class GreedyAllocation:
    def __init__(self, numuser, numRU, H, B, P, RminK, Thrmin, BandW, N0):
        # Khởi tạo các tham số môi trường
        self.numuser = numuser  # Tập người dùng
        self.numRU = numRU  # Tập RU
        self.H = H  # Ma trận channel gain: H[i][k]
        self.B = B  # Danh sách RB của mỗi RU
        self.P = P  # Công suất mỗi RB của RU
        self.RminK = RminK  # Yêu cầu throughput tối thiểu từng user
        self.Thrmin = Thrmin  # Tổng throughput mục tiêu
        self.BandW = BandW  # Băng thông mỗi RB
        self.N0 = N0  # Noise power density

        # Tạo ma trận phân bổ và công suất: allocation[i][b][k], power[i][b][k]
        self.runtime = 0
    
    def allocate_demand_ratio(self):
        weighted_score = self.RminK / self.H**0.5
        start = time.time()
        allocation = np.zeros((self.numRU, self.numuser))

        for ru in range(self.numRU):
            scores = weighted_score[ru]
            for k in range(self.numuser):
                allocation[ru][k] = int(self.B[ru] * scores[k] / sum(scores))
        end = time.time()
        self.runtime = start - end
        # Tính throughput
        dataRate = np.zeros(self.numuser)
        for ru in range(self.numRU):
            for k in range(self.numuser):
                rb = allocation[ru][k]
                gain = self.H[ru][k]
                rate = rb * self.BandW * np.log2(1 + ((self.P[ru] * gain**2) / (self.BandW * self.N0)))
                dataRate[k] += rate

        return allocation, dataRate

    def evaluate_demand_ratio(self):
        allocation, dataRate = self.allocate_demand_ratio()
        pi = [1 for k in range(self.numuser) if dataRate[k] >= self.RminK[k]]
        num_served_users = int(np.sum(pi))
        throughput = sum(dataRate)
        objective_value = (1 - common.tunning) * (throughput/self.Thrmin) + common.tunning * sum(pi)
        

        return {
            'allocation': allocation,
            'dataRate': dataRate,
            'num_served_users': num_served_users,
            'objective': objective_value,
            'pi': pi,
            'throughput': throughput,
            'time':self.runtime
        }


def greedySolve(numuser, numRU, H, B, P, RminK, Thrmin, BandW, N0):
    prob = GreedyAllocation(numuser, numRU, H, B, P, RminK, Thrmin, BandW, N0)
    greedyInfo = prob.evaluate_demand_ratio()

    return greedyInfo.get('num_served_users'), greedyInfo.get('throughput'), greedyInfo.get('time'), greedyInfo.get('objective')