import numpy as np
import pdb
from scipy.spatial import ConvexHull
import scipy
import os

def write_object_data_to_file(obj_instance, output_folder_path, file_name):
    """
    Writes the 'self.seen_data' and 'len(self.x_data)' values
    from an object to a text file in a specified directory.

    Args:
        obj_instance: An instance of an object with 'seen_data' and 'x_data' attributes.
        output_folder_path: The base path for the new directory (e.g., 'my_output_dir').
    """
    output_file_path = os.path.join(output_folder_path, file_name+'.txt')

    try:
        # Create the directory if it does not exist.
        # `exist_ok=True` prevents an error if the directory already exists.
        os.makedirs(output_folder_path, exist_ok=True)
        print(f"Directory '{output_folder_path}' ensured.")

        # Open the file in write mode ('w').
        # If the file exists, its content will be overwritten.
        # If it doesn't exist, a new file will be created.
        with open(output_file_path, 'w') as f:
            # Write the length of x_data
            f.write(f"len(self.x_data): {len(obj_instance.x_data)}\n")
            # Write the value of seen_data
            f.write(f"self.seen_data: {obj_instance.seen_data}\n")

        print(f"Data successfully written to '{output_file_path}'")

    except OSError as e:
        print(f"Error creating directory or writing file: {e}")
    except AttributeError as e:
        print(f"Error: The object is missing required attributes (seen_data or x_data): {e}")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")

class system(object):
	"""docstring for system"""
	def __init__(self, A, B, w_inf, x0):
		self.A     = A
		self.B     = B
		self.w_inf = w_inf
		self.x 	   = [x0]
		self.u 	   = []
		self.w 	   = []
		self.x0    = x0

		self.w_v = w_inf*(2*((np.arange(2**A.shape[1])[:,None] & (1 << np.arange(A.shape[1]))) > 0) - 1)
		print("Disturbance vertices \n", self.w_v )
		
	def applyInput(self, ut):
		self.u.append(ut)
		self.w.append(np.random.uniform(-self.w_inf,self.w_inf,self.A.shape[1]))
		xnext = np.dot(self.A,self.x[-1]) + np.dot(self.B,self.u[-1]) + self.w[-1]
		self.x.append(xnext)

	def reset_IC_given_x0(self,x0):
		self.x = [x0]
		self.u = []
		self.w = []
		
	def reset_IC(self):
		self.x = [self.x0]
		self.u = []
		self.w = []

def dlqr(A, B, Q, R, verbose = False):
	# solve the ricatti equation
	P = np.matrix(scipy.linalg.solve_discrete_are(A, B, Q, R))
	# compute the LQR gain
	K   = np.array(scipy.linalg.inv(B.T*P*B+R)*(B.T*P*A))
	Acl = A - np.dot(B, K)

	if verbose == True:
		print("P: ", P)
		print("K: ", K)
		print("Acl: ", Acl)
	return P, K, Acl
