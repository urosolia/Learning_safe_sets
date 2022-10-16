from casadi import *
from numpy import *
import pdb
import itertools
import numpy as np
import time
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import ConvexHull
from cvxopt import matrix, solvers
import pypoman
import scipy
solvers.options['show_progress'] = False

class IterativeMPC(object):

	def __init__(self, N, A, B, Q, R, Qf, bx, bu, verticesW, opt = 0):
		# Define variables
		self.opt = opt # 0 = worst case cost, 1 = expected cost


		# Data roll-outs current iteration
		self.xData     = [] # list of stored closedLoop
		self.uData     = [] # list of stored input
		self.cData     = [] # list of stored cost

		# Data all iterations
		self.xItData = [] 
		self.uItData = []
		self.cItData = []

		self.robuApp   = [] # each element is a list of inner approximation to the robust reachable sets
		self.costApp   = [] # each element is a list which contains the support point of a pice-wise affine approximation to the value function
		
		self.SS   = []
		self.vApp = []

		self.A = A
		self.B = B
		self.N  = N
		self.n  = A.shape[1]
		self.d  = B.shape[1]
		self.bx = bx
		self.bu = bu
		self.Q = Q
		self.R = R

		self.itCounter = 0
		
		self.verticesW = verticesW
		self.computeRobutInvariant()
		self.A_O, self.b_O = pypoman.duality.compute_polytope_halfspaces(self.verticesO)
		self.A_W, self.b_W = pypoman.duality.compute_polytope_halfspaces(self.verticesW)


	def addData(self, x, u):
		# x = closed-loop trajectory. Rows = time, columns = state coordinate
		# u = closed-loop trajectory. Rows = time, columns = input coordinate
		self.xData.append(x)
		self.uData.append(u)
		self.cData.append( self.computeCost(x,u) )
		self.maxTime = np.array(self.xData).shape[1] # Here assume that all trjectory have same length

	def computeCost(self, x, u):
		# Compute the csot of the roll-out (sum realized cost over closed-loop trajectory)
		for i in range(0, x.shape[0]):
			idx = x.shape[0] - 1 - i 
			xt = x[idx,:]
			if i == 0:
				c = [self.evalH(xt, -np.dot(self.K,xt))]
			else:
				ut = u[idx,:]
				c.append(c[-1] + self.evalH(xt,ut))

		costVector = np.array([np.flip(c)]).T

		return costVector

	def buildTerminalComponents(self):
		print("Start building terminal components")
		# Format data 
		xDataCurrRollOut = [data for data in self.xData]
		uDataCurrRollOut = [data for data in self.uData]
		cDataCurrRollOut = [data for data in self.cData]
		
		self.xItData.append(xDataCurrRollOut)
		self.uItData.append(uDataCurrRollOut)
		self.cItData.append(cDataCurrRollOut)
		
		self.xData = []
		self.uData = []
		self.cData = []

		pruneDate = 0 # 0 = evaluate cost at all stored data points, 1 = evaluate cost only at the vertices

		# Approximate robust reachable sets
		# SS_curr_it: is a list of lists, where each list is a collection of vertices which approximate the robust reachable sets
		SS_curr_it = [np.array(self.xItData[self.itCounter])[0:2,0,:]] # at t = 0 simply pick two data points (ICs are identical at all iterations)
		for t in range(1, self.maxTime):
			# At each t approximate the robust reachable sets with the convex hull of the data collected at time t
			if pruneDate == 0:
				SS_curr_it.append(np.array(self.xItData[self.itCounter])[:,t,:])
			else:
				vertices = np.array(self.xItData[self.itCounter])[:,t,:]
				cvxHull = ConvexHull(vertices)
				extremes = []
				for idx in cvxHull.vertices:
					extremes.append(vertices[idx])
				SS_curr_it.append( np.array(extremes) ) # list of robust reachables sets approximated at that time instant

		self.robuApp.append(SS_curr_it)

		# Approximate value function
		supportPoints_curr_it = []
		for t in range(0, self.maxTime):
			# Hyperplane Cost = ax + b by solving an optimization problem, which upper bounds the cost of the roll-outs
			[a, b] = self.computeHyperplane(t, np.array(self.xItData[self.itCounter])[:,t,:], np.array(self.cItData[self.itCounter])[:,t,:])
			costAtExtremes = []
			for extreme in SS_curr_it[t]:
				# Evaluate Hyperplane Cost at the extremes of the safe set
				costAtExtremes.append(np.dot(a, extreme)+b)
			supportPoints_curr_it.append(np.array(costAtExtremes))

		self.costApp.append(supportPoints_curr_it)
		
		# All the above was for one iteration. Then, we combine with old data
		# Update SS
		xSS = np.vstack(self.robuApp[self.itCounter]) # state to add to SS
		cSS = np.hstack(self.costApp[self.itCounter]) # cost associated with these states
		xcData = np.hstack((xSS,np.array([cSS]).T))   # big matrix of state and cost

		if self.itCounter >0:
			# Merge old safe set and terminal cost function with new data
			xcDataPrevSS = np.hstack((self.SS[self.itCounter-1],np.array([self.vApp[self.itCounter-1]]).T))
			xcData = np.vstack((xcDataPrevSS, xcData))
		else:
			# Merge invariant with new data (the cost is zero if in the invariant)
			xcVectices = np.hstack((self.verticesO, np.zeros((self.verticesO.shape[0],1))))
			xcData = np.vstack((xcVectices, xcData))

		# Do cvx one more time
		cvxHull = ConvexHull(xcData)
		
		extremes = []
		for idx in cvxHull.vertices:
			extremes.append(xcData[idx])

		matrixExtremes = np.vstack(np.array(extremes))
		self.SS.append(matrixExtremes[:, 0:xSS.shape[1]])
		self.vApp.append(matrixExtremes[:, -1])
		# Update FTOCP
		self.buildFTOCP()

		# Update iteration counter
		print("Done building terminal components")
		stateOut = self.robuApp[self.itCounter]
		targets  = self.costApp[self.itCounter]
		self.itCounter += 1
		return stateOut, targets

	def computeHyperplane(self, t, x, c):
		# Compute a hyperplane which approximates the value function over the approximation to the robust reachable sets
		if t == 0:
			a = np.zeros(x.shape[1])
			b = np.max(c)
		else:
			M = np.append(x, np.ones((x.shape[0],1)), axis=1)
			# Cost matrices
			Q =  matrix( np.dot(M.T, M) )
			b = -matrix(np.dot(c.T, M)).T
			# # Inequality constraints
			# G = -matrix( M )
			# h = -matrix(c)
			# Solve
			# sol = solvers.qp(Q, b, G, h)
			sol = solvers.qp(Q, b)
			a = np.squeeze(np.array(sol['x'][0:x.shape[1]]))
			b = np.array(sol['x'][-1])
		return a, b

	def solve(self, x0, verbose=False):
		# Solve the FTOCP
		# Set initial condition + state and input box constraints
		self.lbx = x0.tolist() + (-self.bx).tolist()*(self.N) + (-self.bu).tolist()*self.N + [0]*self.SSdim + (-self.bx).tolist()*(self.N)
		self.ubx = x0.tolist() + ( self.bx).tolist()*(self.N) + ( self.bu).tolist()*self.N + [1]*self.SSdim + ( self.bx).tolist()*(self.N)

		# Solve nonlinear programm
		start = time.time()
		sol = self.solver(lbx=self.lbx, ubx=self.ubx, lbg=self.lbg_dyanmics, ubg=self.ubg_dyanmics)
		end = time.time()
		self.solverTime = end - start
		print("solver time: ", self.solverTime)

		# Check if the solution is feasible
		if (self.solver.stats()['success']):
			self.feasible = 1
			x = sol["x"]
			self.xPred = np.array(x[0:(self.N+1)*self.n].reshape((self.n,self.N+1))).T
			self.uPred = np.array(x[(self.N+1)*self.n:((self.N+1)*self.n + self.d*self.N)].reshape((self.d,self.N))).T
			self.lPred = np.array(x[((self.N+1)*self.n + self.d*self.N):((self.N+1)*self.n + self.d*self.N+self.SSdim)])
			self.dPred = np.array(x[((self.N+1)*self.n + self.d*self.N+self.SSdim):((self.N+1)*self.n + self.d*self.N+self.SSdim+self.n*self.N)].reshape((self.n,self.N))).T

			self.mpcInput = self.uPred[0][0]
		else:
			self.xPred = np.zeros((self.N+1,self.n) )
			self.uPred = np.zeros((self.N,self.d))
			self.mpcInput = []
			self.feasible = 0
			print("Unfeasible")
			
		return self.uPred[0]

	def buildFTOCP(self):
		# Build the probem
		# The function is ||x||_{\mathcal{O}} = \min_{d \in \mathcal{O}} ||x-d||_2

		# Define variables
		n  = self.n
		d  = self.d

		# Define variables
		X      = SX.sym('X', n*(self.N+1));
		U      = SX.sym('U', d*self.N);
		self.SSdim = self.SS[self.itCounter].shape[0]
		lamb   = SX.sym('lamb', self.SSdim);

		# Define the variable D, which is used to compute the d
		D      = SX.sym('D', n*(self.N)); # (X - D) = distance from the invariant

		# Define dynamic constraints
		self.constraint = []
		for i in range(0, self.N):
			X_next = self.dynamics(X[n*i:n*(i+1)], U[d*i:d*(i+1)])
			for j in range(0, self.n):
				self.constraint = vertcat(self.constraint, X_next[j] - X[n*(i+1)+j] ) 

		# terminal constraints
		# x_{N} = {matrix of data}\in R^{n \times number of data poins} \times {vector of multipliers lamb} 
		self.constraint = vertcat(self.constraint, X[n*self.N:(n*(self.N+1))] - mtimes( self.SS[self.itCounter].T ,lamb) )
		# to enfoce cvx: 1 = {vector of ones} \times lamb
		self.constraint = vertcat(self.constraint, 1 - mtimes(np.ones((1, self.SSdim )), lamb ) )

		# The goal set \mathcal{O} = {d \in R^n : self.A_O d <= self.b_O}
		for i in range(0, self.N):
			self.constraint = vertcat(self.constraint, self.A_O @ D[n*i:n*(i+1)] ) 


		# Defining Cost (We will add stage cost later)
		self.cost = 0
		for i in range(0, self.N):
			# self.cost = self.cost + (X[n*i:n*(i+1)]).T @ self.Q @ (X[n*i:n*(i+1)])
			self.cost = self.cost + (X[n*i:n*(i+1)]   -    D[n*i:n*(i+1)]).T @ self.Q @ (X[n*i:n*(i+1)]  -   D[n*i:n*(i+1)])
			self.cost = self.cost + (U[d*i:d*(i+1)]+self.K@D[n*i:n*(i+1)]).T @ self.R @ (U[d*i:d*(i+1)]+self.K@D[n*i:n*(i+1)])

		# terminal cost
		self.cost = self.cost + mtimes(np.array([self.vApp[self.itCounter]]), lamb)

		# Set IPOPT options
		# opts = {"verbose":False,"ipopt.print_level":0,"print_time":0,"ipopt.mu_strategy":"adaptive","ipopt.mu_init":1e-5,"ipopt.mu_min":1e-15,"ipopt.barrier_tol_factor":1}#, "ipopt.acceptable_constr_viol_tol":0.001}#,"ipopt.acceptable_tol":1e-4}#, "expand":True}
		opts = {"verbose":False,"ipopt.print_level":0,"print_time":0}#\\, "ipopt.acceptable_constr_viol_tol":0.001}#,"ipopt.acceptable_tol":1e-4}#, "expand":True}
		nlp = {'x':vertcat(X, U, lamb, D), 'f':self.cost, 'g':self.constraint}
		# nlp = {'x':vertcat(X, U, lamb), 'f':self.cost, 'g':self.constraint}
		self.solver = nlpsol('solver', 'ipopt', nlp, opts)

		# Set lower bound of inequality constraint to zero to force n*N state dynamics + (n+1) for terminal cnstr
		self.lbg_dyanmics = [0]*(n*self.N) + [0]*(n+1) + [-100000]*self.b_O.shape[0]*self.N
		self.ubg_dyanmics = [0]*(n*self.N) + [0]*(n+1) + self.b_O.tolist()*self.N

	def dynamics(self, x, u):
		return self.A @ x + self.B @ u

	def evalH(self, x, u):
		# Evaluate the following cost function:
		# h(x,u) = (x-d)^T Q (x-d) + (u - (-Kd) ) ^T R (u - (-Kd) )
		# h(x,u) is a scaled distance to the invaraiant O and it is computed solving a QP

		# Build matrices
		# DO TO: change cost
		Mx = np.dot(self.Q**0.5, self.verticesO.T)
		Mu = np.dot(self.R**0.5, np.dot(-self.K, self.verticesO.T))
		x = np.dot(self.Q**0.5, x)
		u = np.dot(self.R**0.5, u)

		# cost
		Q =  matrix( np.dot(Mx.T, Mx) ) + matrix( np.dot(Mu.T, Mu) )
		p =  -matrix(np.dot(x.T, Mx)) - matrix(np.dot(u.T, Mu))
		# inequality
		G = matrix( np.vstack((-np.eye(Mx.shape[1]), np.eye(Mx.shape[1])) ) )
		h = matrix( np.hstack((np.zeros(Mx.shape[1]), np.ones(Mx.shape[1]))))
		# equality
		A = matrix( np.ones(Mx.shape[1]) ).T
		b = matrix(1.0)

		# solve and compute cost
		sol = solvers.qp(Q, p, G, h, A, b)
		lamb = sol['x']
		diff_x = np.dot( self.Q**0.5, x - np.squeeze(np.dot(Mx, lamb)) )
		diff_u = np.dot( self.R**0.5, u - np.squeeze(np.dot(Mu, lamb)) )
		cost = np.dot(diff_x, diff_x) + np.dot(diff_u, diff_u) 
		return cost

	def computeRobutInvariant(self):
		self.O_v = [np.array([0,0])]
		self.dlqr()
		print("Compute robust invariant")
		# TO DO:
		# - add check for convergence
		# - add check for input and state constraint satifaction
		for i in range(0,20):
			self.O_v = self.MinkowskiSum(self.O_v, self.verticesW.tolist())

		self.verticesO = np.array(self.O_v)

	def MinkowskiSum(self, setA, setB):
		vertices = []
		for v1 in setA:
			for v2 in setB:
				vertices.append(np.dot(self.Acl,v1) + v2)

		cvxHull = ConvexHull(vertices)
		verticesOut = []
		for idx in cvxHull.vertices:
			verticesOut.append(vertices[idx])

		return verticesOut

	def dlqr(self):
		# solve the ricatti equation
		P = np.matrix(scipy.linalg.solve_discrete_are(self.A, self.B, self.Q, self.R))
		# compute the LQR gain
		self.K   = np.array(scipy.linalg.inv(self.B.T*P*self.B+self.R)*(self.B.T*P*self.A))
		self.Acl = self.A - np.dot(self.B, self.K)

	def plotIt(self, it =-1):
		# plot the approximated reachable set and the value function approximation for iteration it

		# if not speficied pick the last iteration
		if it == -1: it = self.itCounter-1

		# Format data
		dataArray = []
		for listData in [self.xItData[it], self.uItData[it], self.cItData[it]]:
			data = np.array(listData)
			data = data.reshape((data.shape[0]*data.shape[1],data.shape[2]))
			dataArray.append(data)

		# Plot
		plt.figure()		
		ax = plt.axes(projection='3d')
		plt.plot(self.verticesO[:,0], self.verticesO[:,1], 'or-')
		# ax.scatter3D(dataArray[0][:,0], dataArray[0][:,1], dataArray[2], cmap='Reds');
		ax.scatter(dataArray[0][:,0], dataArray[0][:,1], dataArray[2], marker='.', c='k');

		plt.plot(self.robuApp[it][0][0,0], self.robuApp[it][0][0,1], '.r-', lw=2, label='extreme points' )
		for robReachApp in self.robuApp[it]:
			plt.plot(robReachApp[:,0], robReachApp[:,1], 'r-')

		for idx in range(0, len(self.robuApp[it])):		
			verticesSafeSet = self.robuApp[it][idx]
			supportPoitns   = self.costApp[it][idx]
			ax.plot(verticesSafeSet[:,0], verticesSafeSet[:,1], supportPoitns, marker='.', c='b');

		plt.title('Iteration: %i' %it)		
		plt.legend()

		plt.figure()
		plt.plot(self.verticesO[:,0], self.verticesO[:,1], 'or-')
		plt.plot(self.robuApp[it][0][0,0], self.robuApp[it][0][0,1], 'or-', lw=2, label='extreme points' )
		for robReachApp in self.robuApp[it]:
			plt.plot(robReachApp[:,0], robReachApp[:,1], 'or-')
		plt.plot(dataArray[0][:,0], dataArray[0][:,1], '.k', label='stored data')
		plt.title('Iteration: %i' %it)
		plt.legend()
