import numpy as np
from utils import system, dlqr
import pdb
import matplotlib.pyplot as plt
from iterativempc import IterativeMPC
from mpc import MPC
from matplotlib import rc
from build_robust_invariant import BuildRobustInvariant
# rc('font',**{'family':'sans-serif','sans-serif':['Helvetica']})
# rc('text', usetex=True)

# =============================
# Initialize system's parameters
A = np.array([[1, 1],
	          [0, 1]]);
B = np.array([[0], 
			  [1]]);

w_inf = 0.05                    # infinity norm of the disturbance
x0    = np.array([-0.0, 0.0])   # initial condition

alpha      = 3
sys        = system(A, B, w_inf, x0) # initialize system object
sys_aug    = system(A, B, alpha * w_inf, x0) # initialize system object

maxRollOut = 50
maxTime    = 40
maxIt      = 10

N_mpc  = 10
Q      = np.eye(2)*0.1
R      = 0.01*np.eye(1)
Qf     = np.eye(2)

# Initialize mpc parameters
bx =  10*np.ones(A.shape[1]) # state box constraints
bu =  1*np.ones(B.shape[1])  # input box constraints

mpc = MPC(N_mpc, A, B, Q*100, R, Qf, bx, bu)
verticesW = sys.w_v
# impc = IterativeMPC(N_impc, A, B, Q, R, Qf, bx, bu, verticesW)
# verticesO  = impc.verticesO
build_robust_invariant = BuildRobustInvariant(A, B, Q, R, bx, bu, verticesW, alpha)

P, K, Acl = dlqr(A, B, Q, R)
# =============================
# Compute robust invariant from data
x_cl = []; u_cl = [];

for it in range(0,maxIt):
	for rollOut in range(0, maxRollOut): # Roll-out loop
		sys_aug.reset_IC() # Reset initial conditions
		print("Start roll out: ", rollOut, " of iteration: ", it)
		for t in range(0,maxTime): # Time loop
			# ut = mpc.solve(sys.x[-1])
			ut = -np.dot(K, sys_aug.x[-1])
			sys_aug.applyInput(ut)

		# Closed-loop trajectory. The convention is row = time, col = state
		x_cl.append(np.array(sys_aug.x))
		u_cl.append(np.array(sys_aug.u))
		build_robust_invariant.add_data(sys_aug.x)
		# TO DO: check if trajectory reached the goal
		# impc.addData(x_cl[-1], u_cl[-1]) # Add data while performing the task
	if build_robust_invariant.check_robust_invariance():
		print("Robust invariant found")
		break
	else:
		print("Robust invariant NOT found")
# TO DO: only need to change cost!!!
# print(state, targets)

# # =============================
# Plotting 
plt.figure()
for it in range(0, len(x_cl)):
	plt.plot(x_cl[it][:,0], x_cl[it][:,1], 'sb')
		
plt.plot(build_robust_invariant.verticesO[:,0], build_robust_invariant.verticesO[:,1], 'or-')
plt.plot(np.array(build_robust_invariant.x_data)[:,0], np.array(build_robust_invariant.x_data)[:,1], '*k')
for x in build_robust_invariant.x_data:
	for w in build_robust_invariant.verticesW:
		x_next = np.dot(build_robust_invariant.Acl,x) + w
		plt.plot(x_next[0], x_next[1], '*g')
		
plt.xlabel('$x_1$')
plt.ylabel('$x_2$')

# # print("Done with sim. Plotting Starts")
# plt.figure()
# for i in range(len(impc.cItData)):
# 	realizedCost = np.array(impc.cItData[i])[:,0,:]
# 	plt.plot([i]*realizedCost.shape[0], realizedCost, 'or-')
# plt.xlabel('iteration')
# plt.ylabel('cost')

# # Closed-loop across iterations (a bit messy now)
# # plt.figure()
# # plt.plot(verticesO[:,0], verticesO[:,1], 'or-')
# # for it in range(0, len(x_cl)):
# # 	plt.plot(x_cl[it][:,0], x_cl[it][:,1], 's')
# # plt.title('closed-loop')
# # plt.xlabel('$x_1$')
# # plt.ylabel('$x_2$')

# # plt.figure()
# # for it in range(0, len(u_cl)):
# # 	plt.plot(u_cl[it], 'o--')
# # plt.title('input')
# # plt.xlabel('time')
# # plt.ylabel('u')


# # plot value functuin assiciated with one iteration (different from total value function)
# impc.plotIt(it = 0)
# impc.plotIt(it = 1)
# impc.plotIt()
plt.show()