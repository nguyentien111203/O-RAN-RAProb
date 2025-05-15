from itertools import product
import numpy as np


class RUAgent:
    def __init__(self, ru_idx, numuser, RminK, B, shared_Q_tables):
        self.ru_idx = ru_idx
        self.numuser = numuser
        self.RminK = RminK
        self.actions = [(i, j) for i in range(numuser) for j in range(numuser) if i != j]
        self.Q_table = shared_Q_tables
        self.B = B

    def get_allocation_states(self):
        """Sinh tất cả các trạng thái phân bổ PRB nguyên cho từng user sao cho tổng không vượt quá B_max"""
        B_max = max(self.B)  # hoặc dùng giá trị điển hình nếu bạn xét 1 RU tại một thời điểm
        allocation_states = []

        def gen_states(num_user, budget, prefix=[]):
            """Đệ quy sinh tất cả các tổ hợp số nguyên không âm có tổng ≤ budget"""
            if num_user == 1:
                if sum(prefix) <= budget:
                    allocation_states.append(tuple(prefix + [budget - sum(prefix)]))
                return
            for i in range(budget + 1 - sum(prefix)):
                gen_states(num_user - 1, budget, prefix + [i])

        gen_states(self.numuser, B_max)
        return allocation_states

    def get_rgap_states(self):
        """Tạo các trạng thái rời rạc cho R_gap"""
        # Với 2 mức : 0 là đã đủ, 1 là đạt
        return list(product([0, 1], repeat=self.numuser))

    def state_to_key(self, state):
        """Chuyển trạng thái thành key cho Q-table"""
        alloc = tuple(state['allocation'])
        rgap = tuple(state['rgap'])
        return (alloc, rgap)

    def choose_action(self, state, epsilon):
        """Chọn hành động theo chính sách ε-greedy"""
        state_key = self.state_to_key(state)
        if state_key not in self.Q_table:
            self.Q_table[state_key] = {a: 0 for a in self.actions}
        # Nếu số được chọn nhỏ hơn epsilon -> chọn hành động bất kỳ
        if np.random.rand() < epsilon:
            return self.actions[np.random.randint(len(self.actions))]
        # Nếu không, chọn hành động có giá trị Q lớn nhất (nếu nhiều hơn 1 thì random)
        q_values = self.Q_table[state_key]
        max_q = max(q_values.values())
        best_actions = [a for a, q in q_values.items() if q == max_q]

        index = np.random.choice(len(best_actions))
        return best_actions[index]

    def update_q_table(self, state, action, reward, next_state, alpha, gamma):
        """Cập nhật Q-table"""
        state_key = self.state_to_key(state)
        next_state_key = self.state_to_key(next_state)
        # Bổ sung state key nếu hiện không có trong bảng Q
        if state_key not in self.Q_table:
            self.Q_table[state_key] = {a: 0 for a in self.actions}
        if next_state_key not in self.Q_table:
            self.Q_table[next_state_key] = {a: 0 for a in self.actions}
        current_q = self.Q_table[state_key][action]
        next_max_q = max(self.Q_table[next_state_key].values())
        # Cập nhật theo công thức
        self.Q_table[state_key][action] = current_q + alpha * (reward + gamma * next_max_q - current_q)