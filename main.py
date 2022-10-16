import numpy as np
from fnc.polytope import polytope
from fnc.utils import system, dlqr
import pdb
import matplotlib.pyplot as plt
from fnc.iterativempc import IterativeMPC
from fnc.mpc import MPC
from matplotlib import rc
from fnc.build_robust_invariant import BuildRobustInvariant
from fnc.build_control_invariant import BuildControlInvariant
rc('font',**{'family':'sans-serif','sans-serif':['Helvetica']})
rc('text', usetex=True)

# # =================================================================================================
# Initialize system's parameters, constraint sets, and cost function matrices
A = np.array([[1, 1],
	          [0, 1]]);
B = np.array([[0], 
			  [1]]);

barA = np.array([[0.995, 1],
				[0, 0.995]]);
# barA = np.array([[1, 1],
# 	          [0, 1]]);
barB = np.array([[0], 
			  	[1]]);

w_inf = 0.1  # infinity norm of the disturbance
e_inf = 0.05  # infinity norm of the disturbance

alpha = 3


bx =  []
bx.append( 10*np.ones(A.shape[1])) # max state box constraints
bx.append(-10*np.ones(A.shape[1])) # min state box constraints

bu = []
bu.append(2.0*np.ones(B.shape[1]))  # max input box constraints
bu.append(-2.0*np.ones(B.shape[1]))  # max input box constraints

Q      = np.eye(2)
R      = 0.1*np.eye(1)

# =================================================================================================
# Compute robust invariant from data
maxRollOut = 50
maxTime    = 40
maxIt      = 20

x0    = np.array([-0.0, 0.0])   # initial condition
sys        = system(A, B, w_inf, x0) # initialize system object
sys_with_e = system(A, B, w_inf+e_inf, x0) # initialize system object
sys_aug    = system(A, B, alpha * (w_inf + e_inf), x0) # initialize system object

verticesW = sys_with_e.w_v # Vertices of true disturbance
build_robust_invariant = BuildRobustInvariant(barA, barB, A, B, Q, R, bx, bu, verticesW, maxIt, maxTime, maxRollOut, sys_aug)
build_robust_invariant.build_robust_invariant()

# =================================================================================================
# MPC with robust invariant \hat{\mathcal{O}}^r as terminal constraint
N  = 5
Qf = np.eye(2)

# Initialize mpc parameters
build_robust_invariant.shrink_constraint()
mpc = MPC(N, barA, barB, Q, R, Qf, 
			build_robust_invariant.bx_shrink, 
			build_robust_invariant.bu_shrink, 
			build_robust_invariant.K, 
			np.array(build_robust_invariant.x_data),
			np.array(build_robust_invariant.x_data)
			)

x0    = np.array([-9, 1.0])   # initial condition
sys        = system(A, B, w_inf, x0) # initialize system object

# =============================
# Compute control invariant from data
x_cl = []; u_cl = [];
maxRollOut = 50
maxTime    = 10
maxIt      = 10

build_control_invariant = BuildControlInvariant(barA, barB, maxIt, maxTime, maxRollOut,
												 sys, mpc, 
												 build_robust_invariant.x_data.copy(),
												 build_robust_invariant.u_data.copy(),
												 store_all_data_flag = True)

build_control_invariant.build_control_invariant()

# =================================================================================================
# MPC with robust invariant \hat{\mathcal{O}}^r as terminal constraint
N  = 5
Qf = np.eye(2)

# Initialize mpc parameters
mpc_r = MPC(N, barA, barB, Q, R, Qf, 
			build_robust_invariant.bx_shrink, 
			build_robust_invariant.bu_shrink, 
			build_robust_invariant.K, 
			np.array(build_robust_invariant.x_data),
			np.array(build_control_invariant.x_data)
			)

x0_new    = np.array([-10, 1.0])   # initial condition
sys        = system(A, B, w_inf, x0_new) # initialize system object

maxRollOut = 50
maxTime = 10

x_cl = []; u_cl = []
for rollOut in range(0, maxRollOut): # Roll-out loop
	sys.reset_IC() # Reset initial conditions
	print("Start roll out: ", rollOut)
	for t in range(0,maxTime): # Time loop
		ut = mpc_r.solve(sys.x[-1])
		sys.applyInput(ut)

	# Closed-loop trajectory. The convention is row = time, col = state
	x_cl.append(np.array(sys.x))
	u_cl.append(np.array(sys.u))
	
# =============================

# =================================================================================================
# Plotting 
if A.shape[0]==2:
	plt.figure()
	for it in range(0, len(build_robust_invariant.x_cl)):
		if it == 0:
			plt.plot(build_robust_invariant.x_cl[it][:,0], build_robust_invariant.x_cl[it][:,1], '*b', label='Historical data')
		else:
			plt.plot(build_robust_invariant.x_cl[it][:,0], build_robust_invariant.x_cl[it][:,1], '*b')
	minimal_RPI = polytope(vertices=build_robust_invariant.verticesO)
	data_RPI = polytope(vertices=np.array(build_robust_invariant.x_data))
	minimal_RPI.plot2DPolytope('r', linestyle='-*', label = 'Minimal RPI')
	data_RPI.plot2DPolytope('k', linestyle='-o', label = 'RPI from Algorithm 1')
	# plt.plot(np.array(build_robust_invariant.x_data)[:,0], np.array(build_robust_invariant.x_data)[:,1], '*k', label='Historical data')
	plt.legend()
	plt.savefig('invariant.pdf')


plt.figure()
for it in range(0, len(build_robust_invariant.x_cl)):
	plt.plot(build_robust_invariant.x_cl[it][:,0], build_robust_invariant.x_cl[it][:,1], 'sb')
		
plt.plot(build_robust_invariant.verticesO[:,0], build_robust_invariant.verticesO[:,1], 'or-')
plt.plot(np.array(build_robust_invariant.x_data)[:,0], np.array(build_robust_invariant.x_data)[:,1], '*k')
for x in build_robust_invariant.x_data:
	for w in build_robust_invariant.verticesW:
		x_next = np.dot(build_robust_invariant.Acl,x) + w
		plt.plot(x_next[0], x_next[1], '*g')
		
data_RPI = polytope(vertices=np.array(build_robust_invariant.x_data))
data_RPI.plot2DPolytope('k', linestyle='-o', label = 'RPI from Algorithm 1')
plt.xlabel('$x_1$')
plt.ylabel('$x_2$')
# for it in range(0,maxIt):
# 	for rollOut in range(0, maxRollOut): # Roll-out loop
# 		sys.reset_IC() # Reset initial conditions
# 		print("Start roll out: ", rollOut, " of iteration: ", it)
# 		for t in range(0,maxTime): # Time loop
# 			ut = mpc.solve(sys.x[-1])
# 			sys.applyInput(ut)
# 		# Closed-loop trajectory. The convention is row = time, col = state
# 		x_cl.append(np.array(sys.x))
# 		u_cl.append(np.array(sys.u))

# =================================================================================================
# Plotting 
plt.figure()
# for it in range(0, len(build_robust_invariant.x_cl)):
# 	plt.plot(build_robust_invariant.x_cl[it][:,0], build_robust_invariant.x_cl[it][:,1], 'sb')
		
# plt.plot(build_robust_invariant.verticesO[:,0], build_robust_invariant.verticesO[:,1], 'or-')
# for x in build_robust_invariant.x_data:
# 	for w in build_robust_invariant.verticesW:
# 		x_next = np.dot(build_robust_invariant.Acl,x) + w
# 		plt.plot(x_next[0], x_next[1], '*g')
		

if build_control_invariant.store_all_data_flag == True:
	for x in build_control_invariant.x_cl_data:
		plt.plot(x[:,0], x[:,1], '-or')

plt.plot(np.array(build_control_invariant.x_data)[:,0], np.array(build_control_invariant.x_data)[:,1], 'sb')
plt.plot(np.array(build_robust_invariant.x_data)[:,0], np.array(build_robust_invariant.x_data)[:,1], '*k')

if build_control_invariant.store_all_data_flag == True:
	for x in build_control_invariant.x_cl_data:
		plt.plot(x[-1,0], x[-1,1], 'sg')

for x in x_cl:
	plt.plot(x[:,0], x[:,1], '-*b')

plt.xlabel('$x_1$')
plt.ylabel('$x_2$')

if A.shape[0]==2:
	plt.figure()
	for it in range(0, len(x_cl)):
		if it == 0:
			plt.plot(x_cl[it][:,0], x_cl[it][:,1], '-*b', label='Closed-loop')
		else:
			plt.plot(x_cl[it][:,0], x_cl[it][:,1], '-*b')
	X = polytope(F=np.vstack([np.eye(2), -np.eye(2)]), b=np.hstack([bx[0], -bx[1]]))
	CS = polytope(vertices=np.array(build_control_invariant.x_data))
	X.plot2DPolytope('k', linestyle='-', label = 'Constraint Set')
	CS.plot2DPolytope('r', linestyle='-ob', label = '$\mathcal{CS}^r$')
	# plt.plot(np.array(build_robust_invariant.x_data)[:,0], np.array(build_robust_invariant.x_data)[:,1], '*k', label='Historical data')
	plt.plot(x0[0], x0[1], 'sg', label='Initial condition for policy $\kappa$')
	plt.plot(x0_new[0], x0_new[1], 'om', label='Initial condition policy $\pi^{{MPC}, r}$')
	plt.legend()
	plt.xlabel('$x_1$')
	plt.ylabel('$x_2$')
	ax = plt.gca()
	ax.set_xlim([-11, 3])
	ax.set_ylim([-4, 4])
	plt.savefig('closed_loop.pdf')

plt.show()

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
