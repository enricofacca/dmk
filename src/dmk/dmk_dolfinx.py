# import Solver 
#from rcis import Solver
from copy import deepcopy as cp

import sys

import importlib

import numpy as np
import scipy as sp
import scipy.sparse.linalg as splinalg
from scipy.linalg import norm 
import time as cputiming
import os
#from .linear_solvers import info_linalg


import dolfinx as df
import ufl
from ufl import dx, grad, inner, dot
from dolfinx import mesh as mesh_tools


from petsc4py import PETSc as p4py_PETSc
from mpi4py import MPI


# function operations
#from dolfinx import derivative
#from dolfinx import grad
#from dolfinx import dot
#from dolfinx import dx
#from dolfinx import assemble


from linear_algebra import transpose

from petsc4py import PETSc as p4pyPETSc

class SpaceDiscretization:
    """
    Class containg fem discretization variables
    """
    def __init__(self,mesh):
       #tdens_fem='DG0',pot_fem='P1'):
       """
       Initialize FEM spaces used to discretized the problem
       """
       self.mesh = mesh
       
       # For Pot unknow, create fem, function space, trial and test functions 
       #self.pot_fem = df.FiniteElement(mesh, ('Crouzeix-Raviart',1))
       self.pot_fem_space = df.fem.FunctionSpace(mesh, ('Crouzeix-Raviart',1))
       self.pot_trial = ufl.TrialFunction(self.pot_fem_space)
       self.pot_test = ufl.TestFunction(self.pot_fem_space)
       
       
       # For Tdens unknow, create fem, function space, test and trial
       # space
       #self.tdens_fem = df.FiniteElement('DG', mesh.ufl_cell(), 0)
       self.tdens_fem_space = df.fem.FunctionSpace(mesh, ('DG',0))
       self.tdens_trial = ufl.TrialFunction(self.tdens_fem_space)
       self.tdens_test = ufl.TestFunction(self.tdens_fem_space)

       # create mass matrix $M_i,j=\int_{\xhi_l,\xhi_m}$ with
       # $\xhi_l$ funciton of Tdens
       a = df.fem.form (self.tdens_trial * self.tdens_test*ufl.dx)
       self.tdens_mass_matrix = df.fem.assemble_matrix(a)

       print( self.tdens_mass_matrix)
       # create the mixed function space
       #self.pot_tdens_fem_space = df.FunctionSpace(mesh, self.tdens_fem * self.tdens_fem)

    def build_stiff(self,conductivity):
        """
        Internal procedure to assembly stifness matrix 
        A(tdens)=-Div(\Tdens \Grad)

        Args:
        conductivity: non-negative function describing conductivity

        Returns:
        stiff: PETSC matrix
        """

        u = self.pot_trial
        v = self.pot_test
        a = df.fem.form( conductivity * dot( grad(u), grad(v) ) * dx )

        stiff = df.fem.assemble_matrix(a)
        
        return stiff;


class TdensPotential:
    """ This class contains problem solution tdens and pot such that
    pot solves the P-laplacian solution and 
    $\Tdens=|\Grad \Pot|^{\plapl-2}$
    """
    def __init__(self, problem, tdens0=None, pot0=None):
        """
         Constructor of TdensPotential class, containing unkonws
         tdens, pot
         
         Args:
            
        
        Raise:
        ValueError

        Example:
       
        
        """
        #: Tdens function
        self.tdens = df.fem.Function(problem.tdens_fem_space)
        #self.tdens.project(
        #    df.fem.Constant(problem.mesh,
        #                    p4py_PETSc.ScalarType(1))
        #)
        #print(dir(self.tdens))
        print(dir(self.tdens.vector))
        self.tdens.vector.setArray( p4py_PETSc.ScalarType(1))
        #print(self.tdens.x)
        self.tdens.name="TransportDensity"
        
        #: Potential function
        self.pot = df.fem.Function(problem.pot_fem_space)
        self.pot.vector.setArray( p4py_PETSc.ScalarType(0))
        self.pot.name="Potential"
        
        #: int: Number of tdens variable
        self.n_tdens = self.tdens.vector.size
        #: int: Number of pot variable 
        self.n_pot = self.pot.vector.size

        # define implicitely the velocity
        self.velocity = self.tdens * grad(self.pot) 
        

class PLaplacianProblem:
    """
    This class contains the
    """
    def __init__(self, fems):
        """
        Constructor of problem setup
        """
        self.fems  = fems
        self.p_exponent = 1e15
        self.q_exponent = 1

    def set(self,forcing,p_laplacian=1e15):
        """
        Method to set problem inputs.

        Args:
            rhs (real) : vector on the right-hand side of equation
                         A vel = rhs
            q_exponent (real) : exponent q of the norm |vel|^q
        """
        self.rhs_integrated = df.fem.assemble_vector(
            df.fem.form(forcing*self.fems.pot_test * ufl.dx))
        
        self.p_exponent = p_laplacian
        self.q_exponent = 1 + 1 / (p_laplacian - 1)
        return self

class DmkControls:
    """
    Class with Dmk Solver 
    """
    def __init__(self,
                 deltat=0.5,
                 time_discretization_method='explicit_tdens',
                 approach_linear_solver='bicgstab',
                 max_linear_iterations=1000,
                 tolerance_linear=1e-9,
                 max_nonlinear_iterations=30,
                 tolerance_nonlinear=1e-10):
        """
        Set the controls of the Dmk algorithm
        """
        #: character: time discretization approach
        self.time_discretization_method = time_discretization_method

        #: real: time step size
        self.deltat = deltat

        # variables for set and reset procedure
        self.deltat_control = 0
        self.min_deltat = 1e-2
        self.max_deltat = 1e+2
        self.expansion_deltat = 2
        
        #: int: max number of Krylov solver iterations
        self.max_linear_iterations = max_linear_iterations
        #: str: Krylov solver approach
        self.approach_linear_solver = approach_linear_solver
        #: real: Krylov solver tolerance
        self.tolerance_linear = tolerance_linear
        
        #: real: nonlinear solver iteration
        self.tolerance_nonlinear = tolerance_nonlinear

        #: int: Max number of nonlinear solver iterations 
        self.max_nonlinear_iterations = 20
        
        #: real: minimum newton step
        self.min_newton_step = 5e-2
        self.contraction_newton_step = 1.05
        self.min_C = 1e-6
        
        
        #: Fillin for incomplete factorization
        self.outer_prec_fillin=20
        #: Drop tolerance for incomplete factorization
        self.outer_prec_drop_tolerance=1e-4
        self.relax4prec = 1e-12

        #: info on standard output
        self.verbose=0
        #: info on log file
        self.save_log=0
        self.file_log='admk.log'

# Create a class to store solver info 
class InfoDmkSolver():
    def __init__(self):
        self.linear_solver_iterations = 0
        # non linear solver
        self.nonlinear_solver_iterations = 0
        self.nonlinear_sovler_residum = 0.0

        
class DmkSolver:
    """
    Solver class for problem
    -div(|\Pot|^{\plapl-2}\Grad \Pot)= \Forcing
    
    via Dynamic Monge-Kantorovich.
    We find the long time solution of the
    dynamics 
    \dt \Tdens(t)=\Tdens(t) * | \Grad \Pot(\Tdens)|^2 -Tdens^{gamma}
    
    """
    """
    We extend the class "Solver" ovverriding the the "iterate" procedure. 
    """
    def __init__(self, fems,ctrl=None):
        """
        Initialize solver with passed controls (or default)
        and initialize structure to store info on solver application
        """
        # init controls
        if (ctrl == None):
            self.ctrl = DmkControls()
        else:
            self.ctrl = cp(ctrl)

        # init infos
        self.info = InfoDmkSolver()

        # create pointe to mesh
        self.fems = fems

        
        
    def print_info(self, msg, priority):
        """
        Print messagge to stdout and to log 
        file according to priority passed
        """
        
        if (self.ctrl.verbose > priority):
            print(msg)
        #if (self.ctrl.log >  priority):
        #    print(msg)    
        
   

    def tdens2gfvar(self,tdens,gfvar):
        """
        Transformation from tdens variable to gfvar (gradient flow variable)
        Works only for DG0 discretization
        """
        #if not ( self.fems.tdens_fem.type == 'DG' and
        #         self.fems.tdens_fem.degree == 0):
        #    exit()
        gfvar.vector()[:] = np.sqrt(tdens.vector()[:])

    def gfvar2tdens(self,gfvar,derivative_order,tdens):
        """
        Compute \phi(gfvar)=tdens, \phi' (gfvar), or \phi''(gfvar)
        Works only for DG0 discretization
        """
        if (derivative_order == 0 ):
            tdens.vector()[:] = gfvar.vector()[:] * gfvar.vector()[:]
        elif (derivative_order == 1 ):
            tdens.vector()[:] = 2 * gfvar.vector()[:] 
        elif (derivative_order == 2 ):
            tdens.vectors()[:] = 1.0
        else:
            print('Derivative order not supported')
        return tdens

    def syncronize(self, problem, tdpot,ierr):
        """        
        Args:
         tdpot: Class with unkowns (tdens, pot in this case)
         problem: Class with inputs  (rhs, q_exponent)
         ctrl:  Class with controls

        Returns:
         tdpot : syncronized to fill contraint S(tdens) pot = rhs
         info  : control flag (=0 if everthing worked)
        """

        # assembly stiff
        start_time = cputiming.time()
        stiff = self.fems.build_stiff(tdpot.tdens)

        print(dir(problem.rhs_integrated))
        print(problem.rhs_integrated)
        rhs = df.fem.Function(problem.fems.pot_fem_space)
        print(rhs)
        #rhs = tdpot.pot.copy()
        rhs.array = problem.rhs_integrated


        msg = ('ASSEMBLY'+'{:.2f}'.format(-(start_time - cputiming.time()))) 
        self.print_info(msg,3)        
        
        #
        # solve linear system
        #
        N = problem.rhs_integrated.array.size
        print(dir(mesh_tools))
        mesh1d = mesh_tools.create_unit_interval(MPI.COMM_WORLD,N)
        VI = df.fem.FunctionSpace(mesh1d, ("DG", 0))
        u1d = ufl.TrialFunction(VI)
        v1d = ufl.TestFunction(VI)
        Id = df.fem.assemble_matrix(df.fem.form( N *u1d * v1d * ufl.dx))

        print(dir(stiff))
        stiff.axpy(1e-12,Id,False)

        df.parameters["linear_algebra_backend"] = "PETSc"

        # (algebraic multigrid)
        df.PETScOptions.set("ksp_type", "cg")

        #df.PETScOptions.set("pc_type", "icc")
        #"""
        #MULTIGRID SOLVER
        df.PETScOptions.set("pc_type", "gamg")
        df.PETScOptions.set("mg_coarse_ksp_type", "preonly")
        df.PETScOptions.set("mg_coarse_pc_type", "svd")
        #"""
        # Print PETSc solver configuration
        #df.PETScOptions.set("ksp_view")
        #df.PETScOptions.set("ksp_monitor")
        
        # Set the solver tolerance
        df.PETScOptions.set("ksp_rtol", self.ctrl.tolerance_nonlinear)

        # Create Krylov solver and set operator
        solver = df.PETScKrylovSolver()
        solver.set_operator(stiff)
        solver.set_from_options()
        solver.solve(tdpot.pot.vector(),problem.rhs_integrated) 

        ierr = 0
        
        #return tdpot,ierr,self;

    
    def iterate(self, problem, tdpot, ierr):
        """
        Procedure overriding update of parent class(Problem)
        
        Args:
        problem: Class with inputs  (rhs, q_exponent)
        tdpot  : Class with unkowns (tdens, pot in this case)

        Returns:
         tdpot : update tdpot from time t^k to t^{k+1} 

        """
        print(self.ctrl.time_discretization_method)
        if (self.ctrl.time_discretization_method == 'explicit_tdens'):            
            # compute update
            grad_pot = grad(tdpot.pot)
            
            # integrate w.r.t. to tdens test function
            # -\Tdens | \Grad \Pot|^2 + \Tdens^\pmass
            pmass=problem.q_exponent/(2-problem.q_exponent)
            rhs_ode = assemble(
                ( -tdpot.tdens  * dot(grad_pot,grad_pot) + tdpot.tdens**pmass )
                * self.fems.tdens_test* ufl.dx )
            
            # compute update vectors using mass matrix
            update = df.Function(self.fems.tdens_fem_space)
            df.fem.solving.solve( self.fems.tdens_mass_matrix, update.vector(), rhs_ode) 
            
            # update coefficients of tdens
            tdpot.tdens.vector().axpy(- self.ctrl.deltat, update.vector())
        
            # compute pot associated to new tdens
            self.syncronize(problem,tdpot,ierr)
            
            #return tdpot, ierr, self;
        elif (self.ctrl.time_discretization_method == 'implicit_gfvar'):
        
            # pass in gfvar varaible
            

            gfvar = df.Function(self.fems.tdens_fem_space)
            gfvar_old = df.Function(self.fems.tdens_fem_space)
            self.tdens2gfvar(tdpot.tdens,gfvar_old)
            gfvar.vector()[:] = gfvar_old.vector()[:]
            pot   = df.Function(self.fems.pot_fem_space)
            pot_increment = df.Function(self.fems.pot_fem_space)
            gfvar_increment = df.Function(self.fems.tdens_fem_space)
            

            # two varaible allocated for linesearch
            current_pot = df.Function(self.fems.pot_fem_space)
            current_gfvar = df.Function(self.fems.tdens_fem_space)
            #increment = df.Function(self.fems.pot_tdens_fem_space)
            print(dir(self.fems.tdens_fem_space))
            print(self.fems.tdens_fem_space.dim())
            diag_C = p4pyPETSc.Vec().createWithArray(np.zeros(self.fems.tdens_fem_space.dim()))
            
            
            inewton = 0
            ierr_newton = 0

            # the term u * f is missing
            # because its already integrated
            Energy = (- 0.5* gfvar**2 * dot(grad(pot), grad(pot))
                      + 0.5 * gfvar**2
                      + 0.5 * (gfvar - gfvar_old)**2 ) * ufl.dx

            
            
            # cycle until an error occurs
            while (ierr_newton == 0):
                # assembly nonlinear equation
                # First Gateaux derivative
                F_pot = derivative(Energy, pot)
                F_gfvar = derivative(Energy, gfvar)

                F_pot_vector = assemble(F_pot)
                F_gfvar_vector = assemble(F_gfvar)
                
                # check if convergence is achieved
                self.info.nonlinear_solver_residuum = ( np.linalg.norm(F_pot_vector)
                                                        + np.linalg.norm(F_gfvar_vector) )
                    
                msg=(str(inewton)+
                     ' |F|_pot  = '+'{:.2E}'.format(np.linalg.norm(F_pot_vector)) +
                     ' |F|_gfvar= '+'{:.2E}'.format(np.linalg.norm(F_gfvar_vector)))
                if (self.ctrl.verbose >= -1 ):
                    print(msg)

                print(self.info.nonlinear_solver_residuum)
                
                if ( self.info.nonlinear_solver_residuum < self.ctrl.tolerance_nonlinear ) :
                    ierr_newton == 0
                    break
                
                # assembly jacobian
                self.gfvar2tdens(gfvar,0,tdpot.tdens)
                A_matrix = self.build_stiff(tdpot.tdens)
                BT_matrix = assemble(derivative(F_pot,gfvar) )
                B_matrix = transpose(BT_matrix)
                C_matrix = assemble(derivative(-F_gfvar,gfvar))

                # Create a block matrix
                jacobian = df.BlockMatrix(2, 2)
                jacobian[0, 0] = A_matrix
                jacobian[1, 0] = BT_matrix
                jacobian[0, 1] = B_matrix
                jacobian[1, 1] = -1.0*C_matrix

                a = [[derivative(F_pot,pot),      derivative(F_pot,gfvar)],
                     [derivative(F_gfvar,pot), derivative(-F_gfvar,gfvar)]]

                J = dolfinx.fem.assemble_matrix_nest(a, bcs)
                J.assemble()

                
                # Create a block vector (that is compatible with A in parallel)
                rhs_newton = df.BlockVector(2)
                rhs_newton[0] = F_pot_vector
                rhs_newton[1] = F_gfvar_vector

                # Create a another block vector (that is compatible with A in parallel)
                increment = df.BlockVector(2)
                increment[0] = pot_increment.vector()
                increment[1] = gfvar_increment.vector()

                # class mydf.BlockMatrix(object):
                #      #petsc preconditioner interface
                #     def setUp(self, jacobian):
                #         self.jacobian = jacobian
                #         pass
                #     def apply(self,pc,x,y) 
                
                #ksp = p4pyPETSc.KSP().create()
                ksp = df.PETScKrylovSolver()
                ksp.set_operators(jacobian)
                # Set PETSc solve type (conjugate gradient) and preconditioner
                # (algebraic multigrid)
                df.PETScOptions.set("ksp_type", "cg")
            
                ksp.solve(rhs_newton, increment)
                

                # the minus sign is to get saddle point in standard form
                print(type(A_matrix),type(B_matrix), type(BT_matrix))
                """
                C_matrix.get_diagonal(diag_C)
                msg=('{:.2E}'.format(min(diag_C_matrix))+'<=C <='+'{:.2E}'.format(max(diag_C_matrix)))
                if (self.ctrl.verbose >= 3 ):
                    print(msg)
                inv_C_matrix = sp.sparse.diags(1.0/diag_C_matrix)
                
                # form primal Schur complement S=A+BT * C^{-1} B
                primal_S_matrix = (A_matrix +
                                   BT_matrix * inv_C_matrix * B_matrix
                                   + self.ctrl.relax4prec * sp.sparse.eye(n_pot) )
                
                
                # solve linear system
                # increment
                primal_rhs = ( F_pot_vect + BT_matrix.dot * inv_C_matrix * F_gfvar_vect )

                
                
                increment.vector()[n_pot:n_pot+n_tdens] = - inv_C_matrix.dot(
                    f_newton[n_pot:n_pot+n_tdens] - B_matrix.dot(increment[0:n_pot]))
                
                """
                
                # line search to ensure C being strictly positive
                finished = False
                newton_step = 1.0
                current_pot.vector()[:] = pot.vector()[:]
                current_gfvar.vector()[:] = gfvar.vector()[:]
                print( finished)
                while ( not finished ):
                    print(pot.size())
                    
                    # update pot, gfvar and derived components
                    pot.vector()[:] = ( current_pot.vector()[:]
                                        + newton_step * increment[0].vector()[0:n_pot])
                    
                    gfvar.vector()[:] = ( current_gfvar.vector()[:]
                                          + newton_step * increment[1].vector()[n_pot:n_pot+n_tdens])
                    self.gfvar2tdens(gfvar, 2, trans_second) # 1 means zero derivative so
                    grad_pot = problem.potential_gradient(pot)

                    diag_C_matrix = problem.weight * (
                        1.0 / self.ctrl.deltat
                        + trans_second * 0.5*(-grad_pot **2 + 1.0 )
                    )

                    # ensure diag(C) beingstrctly positive
                    if ( np.amin(diag_C_matrix) < self.ctrl.min_C ):
                        newton_step =  newton_step / self.ctrl.contraction_newton_step
                        if (newton_step < self.ctrl.min_newton_step):
                            print('Newton step=',newton_step,'below limit', self.ctrl.min_newton_step)
                            ierr_newton = 2
                            finished = True
                    else:
                         ierr_newton = 0
                         finished = True


                    print(pot.size())
                msg='Newton step='+str(newton_step)
                if (self.ctrl.verbose >= 3 ):
                    print(msg)
                
                      
                # count iterations
                inewton += 1
                if (inewton == self.ctrl.max_nonlinear_iterations ):
                    ierr_newton = 1
                    # end of newton


           

            # copy the value in tdpot (even if the are wrong)
            tdpot.pot.vector()[:] = pot.vector()[:]
            self.gfvar2tdens(gfvar,0,tdpot.tdens)
            tdpot.time = tdpot.time+self.ctrl.deltat

            # store info algorithm
            self.info.nonlinear_iterations = inewton


            # pass the newton error (0,1,2)
            ierr = ierr_newton
            
        else:
            print('value: self.ctrl.time_discretization_method not supported. Passed:',self.ctrl.time_discretization_method )
            ierr = 1
            #return tdpot, ierr, self;
