import numpy as np
from Model import Model
from matplotlib import pyplot as plt

################################
# TEST model
################################
n = 6  # number of states
m = 4  # number of inputs (Thrusters)
l = 3  # number of outputs

mass = 500         # temp
X_u = 100             # temp
Y_v = 100         # temp
N_r = 100       # temp
Y_r = 10       # temp
N_v = Y_r           # assumption starboard and port symmetries
X_udot = 100       # temp
Y_vdot = 100      # temp
N_rdot = 100       # temp
Y_rdot = 10       # temp
N_vdot = Y_rdot   # assumption starboard and port symmetries
L = 10            # Length
B = 2             # width
T = 1             # depth
rho = 1        # density of water
I_zz = np.pi*rho*L*B*T*(B**2 + L**2)/120
thrusters = [(5, 0, np.pi/2), (5, 0, -np.pi/2), (-0.8*L/2, -B/2, 0), (-0.8*L/2, B/2, 0)]

# Mass matrix
M = np.zeros((3, 3))
M[0, 0] = mass + X_udot
M[1, 1] = mass + Y_vdot
M[1, 2] = Y_rdot
M[2, 1] = N_vdot
M[2, 2] = I_zz + N_rdot

D = np.zeros((3, 3))
D[0, 0] = X_u
D[1, 1] = Y_vdot
D[1, 2] = Y_r
D[2, 1] = N_v
D[2, 2] = N_r

# Initialize state
x0 = np.zeros((n, 1))
x0[2, 0] = np.pi/2

A = np.zeros((n, n))
A[0, 3] = np.round(np.cos(x0[2]), 10)
A[0, 4] = -np.round(np.sin(x0[2]), 10)
A[1, 3] = np.round(np.sin(x0[2]), 10)
A[1, 4] = np.round(np.cos(x0[2]), 10)
A[2, 5] = 1
A[3:6, 3:6] = - np.linalg.inv(M) @ D

B = np.zeros((n, m))
T_thrusters = np.zeros((3, m))
for i, thruster in enumerate(thrusters):
    L_N, L_E, alpha = thruster
    T_thrusters[0, i] = np.round(np.cos(alpha), 10)
    T_thrusters[1, i] = np.round(np.sin(alpha), 10)
    T_thrusters[2, i] = -L_E*np.cos(alpha) + L_N*np.sin(alpha)

B[3:6, :] = np.linalg.inv(M) @ T_thrusters
C = np.zeros((l, n))
C[0, 0] = 1
C[1, 1] = 1
C[2, 2] = 1

dt = 1
prediction_horizon = 500
u = np.zeros((4, prediction_horizon))

u[0, :] = 0
u[1, :] = 0
u[2, :] = 0
u[3, :] = -10
boat_model = Model(x0, A, B, C, prediction_horizon, dt)
prediction = boat_model.calc_prediction(u)

plt.figure()
plt.quiver(prediction[1, :], prediction[0, :], np.sin(prediction[2, :]), np.cos(prediction[2, :]), scale=50)
plt.show()

# plt.plot(prediction[1, :], prediction[0, :])
# plt.title("Location N, E over time")
# plt.show()
#
# plt.plot(prediction[2, :])
# plt.show()
#
plt.plot(prediction[3, :])
plt.title("Forward speed over time")
plt.show()
#
plt.plot(prediction[4, :])
plt.title("Speed to the side")
plt.show()
#
plt.figure()
plt.plot(prediction[2, :])
plt.title("orientation")
plt.show()

plt.figure()
plt.plot(prediction[5, :])
plt.title("Rotational speed around centre")
plt.show()


