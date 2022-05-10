from dmk import SpaceDiscretization
from dmk import DmkControls
from dmk import InfoDmkSolver

from copy import deepcopy as cp


from dolfin import DirichletBC
from dolfin import assemble
from dolfin import dx
from dolfin import grad
from dolfin import dot

from dolfin import Constant



from dolfin import UserExpression
from dolfin import FiniteElement
from dolfin import FunctionSpace
from dolfin import Function
from dolfin import TrialFunction
from dolfin import TestFunction
from dolfin import MixedFunctionSpace

from dolfin.fem.solving import solve


from dolfin import parameters
from dolfin import PETScOptions
from dolfin import PETScKrylovSolver


import time as cputiming


class MinimalSurfaceProblem:
    """
    This class contains the
    """
    def __init__(self, fems):
        """
        Constructor of problem setup
        """
        self.fems  = fems

    def set(self, forcing, boundary_conditions):
        """
        Method to set problem inputs.

        Args:
            rhs (real) : vector on the right-hand side of equation
                         A vel = rhs
            q_exponent (real) : exponent q of the norm |vel|^q
        """
        self.forcing = forcing
        self.bcs=[]
        for bc in boundary_conditions:
            self.bcs.append(DirichletBC(self.fems.pot_fem_space, bc[1], bc[0]))
        self.rhs_integrated = assemble(forcing*self.fems.pot_test*dx)

class MinimalSurfaceSolver:
    """
    Solver class for problem
    
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
        stiff = self.fems.build_stiff(Constant(1.0)/tdpot.tdens)
        for bc in problem.bcs:
            bc.apply(stiff, problem.rhs_integrated)
        
        
        msg = ('ASSEMBLY'+'{:.2f}'.format(-(start_time - cputiming.time()))) 
        self.print_info(msg,3)        
        

        parameters["linear_algebra_backend"] = "PETSc"

        # (algebraic multigrid)
        PETScOptions.set("ksp_type", "cg")

        #PETScOptions.set("pc_type", "icc")
        #"""
        #MULTIGRID SOLVER
        PETScOptions.set("pc_type", "gamg")
        PETScOptions.set("mg_coarse_ksp_type", "preonly")
        PETScOptions.set("mg_coarse_pc_type", "svd")
        #"""
        # Print PETSc solver configuration
        #PETScOptions.set("ksp_view")
        #PETScOptions.set("ksp_monitor")
        
        # Set the solver tolerance
        PETScOptions.set("ksp_rtol", self.ctrl.tolerance_nonlinear)

        # Create Krylov solver and set operator
        solver = PETScKrylovSolver()
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
            rhs_ode = assemble(
                (-( dot(grad_pot,grad_pot) + 1) * Constant(1.0) / tdpot.tdens
                 + tdpot.tdens )
                * self.fems.tdens_test* dx )
            
            # compute update vectors using mass matrix
            update = Function(self.fems.tdens_fem_space)
            solve( self.fems.tdens_mass_matrix, update.vector(), rhs_ode) 
            
            # update coefficients of tdens
            tdpot.tdens.vector().axpy(- self.ctrl.deltat, update.vector())
        
            # compute pot associated to new tdens
            self.syncronize(problem,tdpot,ierr)
            
            #return tdpot, ierr, self;
