import numpy as np


class Model:
    """
    params:
    x0: initial state
    a: A matrix LTI system
    b: B matrix LTI system
    c: C matrix LTI system
    p: prediction horizon
    """
    def __init__(self, x0, a, b, c, p):
        self.current_state = x0
        self.current_time_step = 0
        self.states = []
        self.A = a
        self.B = b
        self.C = c
        self.p = p
    #     self.O = self.gen_o_matrix(p)
    #     self.M = self.gen_m_matrix(p)
    #
    # def gen_o_matrix(self, p):
    #     O = np.zeros((self.C.shape[0]*p, self.A.shape[0]))
    #     pow_a = np.eye(self.A.shape[0])
    #     for i in range(p):
    #         if i == 0:
    #             pow_a = self.A
    #         else:
    #             pow_a = pow_a @ self.A
    #         O[i*self.C.shape[0]:(i+1)*self.C.shape[0], :] = self.C @ pow_a
    #     return O
    #
    # def gen_m_matrix(self, p):
    #     M = np.zeros((self.C.shape[0]*p, self.B.shape[1]))
    #     pow_a = np.eye(self.A.shape[0])
    #     for i in range(p):
    #         if i == 1:
    #             pow_a = self.A
    #         else:
    #             pow_a = np.matmul(pow_a, self.A)
    #         M[i*self.C.shape[0]:(i+1)*self.C.shape[0], :] = self.C @ pow_a @ self.B
    #     return M

    def calc_prediction(self, u):
        x_k = self.current_state
        prediction = []
        for i in range(self.p):
            x_kp1 = self.calc_next_state(x_k, u)
            prediction.append(x_kp1)
            x_k = x_kp1

    def update_state(self, state):
        self.states.append(self.current_state)
        self.current_state = state

    def calculate_next_state(self, x, u):
        self.A[0, 3] = np.cos(x[2])
        self.A[0, 4] = np.sin(-x[2])
        self.A[1, 3] = np.sin(x[2])
        self.A[1, 4] = np.cos(x[2])
        self.A[2, 5] = 1
        return self.A @ x + self.B @ u

    def state_prediction(self):
        return self.O @ self.current_state

