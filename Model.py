import numpy as np


class Model:
    """
    params:
    x0: initial state
    a: A matrix LTI system
    b: B matrix LTI system
    c: C matrix LTI system
    p: prediction horizon
    dt: timestep
    """
    def __init__(self, x0, a, b, c, p, dt):
        self.current_state = x0
        self.current_time_step = 0
        self.states = []
        self.A = a
        self.B = b
        self.C = c
        self.p = p
        self.dt = dt
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
        prediction = np.zeros((self.A.shape[0], self.p))
        for i in range(self.p):
            prediction[:, i:i+1] = x_k
            x_kp1 = self.calculate_next_state(x_k, u[:, i:i+1])
            x_k = x_kp1
        return prediction

    def update_state(self, state):
        self.states.append(self.current_state)
        self.current_state = state

    def calculate_next_state(self, x, u):
        self.A[0, 3] = np.round(np.cos(x[2]), 10)
        self.A[0, 4] = -np.round(np.sin(x[2]), 10)
        self.A[1, 3] = np.round(np.sin(x[2]), 10)
        self.A[1, 4] = np.round(np.cos(x[2]), 10)
        self.A[2, 5] = 1
        x_kp = x + self.A @ x * self.dt + self.B @ u * self.dt
        x_kp[2, 0] = np.mod(x_kp[2, 0], 2*np.pi)
        return x_kp
