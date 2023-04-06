import dolfinx as df
import petsc4py as PETSc

class ControlsLinearAlgebra:
    """
    Class to store information in the linear solvers usage
    """
    def __init__(self,
                 tolerance = 1e-06,
                 max_iterations = 100,
                 verbose = 0,
                 linear_algebra_backend = 'PETSC'):
        self.tolerance = tolerance
        self.max_iterations = max_iterations
        self.verbose = verbose 
        self.linear_algebra_backend = linear_algebra_backend
        


class info_linalg:
    """
    Class to store information in the linear solvers usage
    """
    def __init__(self,linear_algebra_backend='PETSC'):
        self.ierr = 0
        self.iter = 0
        self.resini = 0.0
        self.realres = 0.0
        

    def __str__(self):
        strout=(str(self.info)+' '+
                str(self.iter)+' '+
                str('{:.2E}'.format(self.resini))+' '
                +str('{:.2E}'.format(self.realres)))
        return strout;
    
    def addone(self, xk):
        self.iter += 1

def transpose(A):
    # Using PETSC
    B = A.copy()
    petsc_mat = df.as_backend_type(B).mat()
    petsc_mat.transpose()
    A_adj_petsc = df.PETScMatrix(petsc_mat)
    print(type(A),type(B),type(petsc_mat),type( A_adj_petsc))
    return A_adj_petsc

def dolfin2p4py(A):
    # Using PETSC
    B = A.copy()
    return petsc_mat


