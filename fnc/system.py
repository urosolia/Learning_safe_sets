import numpy as np

def systems_def(opt='double_integrator'):
    # Initialize system's parameters, constraint sets, and cost function matrices
    A = np.array([[1, 1],
                [0, 1]]);
    
    B = np.array([[0], 
                [1]]);




    u_max = 2

    if opt == 'double_integrator':
        x0_mpc    = np.array([-8.0, 1.0])   # initial condition
        x0_lmpc   = np.array([-10, 1.0])   # initial condition
        w_inf = 0.1  # infinity norm of the disturbance
        e_inf = 0.05   # infinity norm of the disturbance
        barA = np.array([[0.95, 1],
                        [0, 0.95]]);
        
        barB = np.array([[0], 
                        [1]]);

        alpha = 1.5

    elif opt == 'double_integrator_2D':
        A = np.kron(np.eye(2), A)
        B = np.kron(np.eye(2), B)

        barA = np.kron(np.eye(2), barA)
        barB = np.kron(np.eye(2), barB)

        x0_mpc    = np.array([-9.0, 1, -9.0, 1])   # initial condition
        x0_lmpc   = np.array([-9.5, 1, -9.5, 1])   # initial condition
        w_inf = 0.01  # infinity norm of the disturbance
        e_inf = 0.005   # infinity norm of the disturbance

        alpha = 10
    elif opt == 'double_integrator_3D':
        A = np.kron(np.eye(3), A)
        B = np.kron(np.eye(3), B)
        A[0, 3] =  0.1
        A[0, 5] = -0.1
        A[2, 5] = 0.1
        barA = np.array([[0.995, 1],
                        [0, 0.995]]);
        
        barB = np.array([[0], 
                        [1]]);

        barA = np.kron(np.eye(3), barA)
        barB = np.kron(np.eye(3), barB)
        barA[0, 3] =  0.1
        barA[0, 5] = -0.1
        barA[2, 5] = 0.1

        x0_mpc    = np.array([-7.5, 1.0, -7.5, 1, -7.5, 1])   # initial condition
        x0_mpc   = np.array([-9.0, 1.0, -9.0, 1, -9.0, 1])   # initial condition
        x0_lmpc   = np.array([-9.0, 1.0, -9.0, 1, -9.0, 1])   # initial condition
        w_inf = 0.01  # infinity norm of the disturbance
        e_inf = 0.005   # infinity norm of the disturbance

        alpha = 10
    return A, B, barA, barB, alpha, w_inf, e_inf, x0_mpc, x0_lmpc, u_max