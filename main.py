import numpy as np
from Model import Model

n = 4  # number of states
m = 1  # number of inputs
l = 1  # number of outputs

A = np.zeros((n, n))
A[0, 1] = 1
A[2, 3] = 1

A[1, 0] = -2
A[1, 1] = -2
A[1, 2] = -1
A[1, 3] = -1

A[3, 0] = 1
A[3, 1] = 1
A[3, 2] = -1
A[3, 3] = -1


B = np.zeros((n, m))
B[3, 0] = 0.1

C = np.zeros((l, n))
C[0, 0] = 1
x0 = np.ones((n, 1))
# print(A @ x0)
# print(x0)
#
# print(A)
# print(B)
# print(C)
boat_model = Model(x0, A, B, C, 9)
print(C)
print(boat_model.state_prediction())
# print(boat_model.O)
# print(boat_model.M)

# boat_model.calculate_next_state(boat_model.current_state, np.zeros((1, 1)))
