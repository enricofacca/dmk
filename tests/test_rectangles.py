import sys
import os
from copy import deepcopy as cp

sys.path.append('../src/dmk')
from dmk import SpaceDiscretization
from dmk import TdensPotential
from dmk import DmkSolver
from dmk import DmkControls
from dmk import PLaplacianProblem

from dolfin import UserExpression
from dolfin import UnitSquareMesh
from dolfin import *

from dolfin import norm


class Rect(UserExpression):
    """
    Define forcing term in readable in fenics assembler
    Test case in
    - Facca et. al 2018 https://doi.org/10.1137/16M1098383
    - Facca et. a. 2021 
    """
    def eval(self, value, x):
        """
        Define Forcing as F(|x,y|)
        """
        value[0]=0.0
        if ( x[1]>=1/4 and x[1]<3/4):
            if ( x[0]>=1/8 and x[0]<=3/8):
                value[0]=2
            if ( x[0]>=5/8 and x[0]<=7/8):
                value[0]=-2

class OptimalTdensRect(UserExpression):
    """Define Optimal Transport density assocaited to 
    forcing term in readable in fenics assembler 
    Test case in 
    - Facca et. al 2018 https://doi.org/10.1137/16M1098383 
    - Facca et. a. 2021

    """
    def eval(self, value, x):
        """
        Define Forcing as F(|x,y|)
        """
        value[0]=0.0
        if ( x[1]>=1/4 and x[1]<3/4):
            if ( x[0]>=1/8 and x[0]<=3/8):
                value[0]=2*(x[0]-1/8)
            if ( x[0]>3/8 and x[0]<5/8):
                value[0]=0.5
            if ( x[0]>=5/8 and x[0]<=7/8):
                value[0]=2*(7/8-x[0])

# create uniform grid of square [0,1]x[0,1]
ndiv = 32
mesh = UnitSquareMesh(ndiv, ndiv)

# set expression for forcing term
forcing=Rect()
optimal_tdens=OptimalTdensRect()


# Define problem disctetization
p1p0 = SpaceDiscretization(mesh)

# Create a class descibing problem inputs given
# a Space Discretization and fill it
mkeqs = PLaplacianProblem(p1p0)
mkeqs.set(forcing)

# Define problem solution
tdpot = TdensPotential(p1p0)

# init iterative solver
ctrl_solver = DmkControls(time_discretization_method='explicit_tdens',
                          tolerance_nonlinear = 1e-6)
dmk_solver = DmkSolver(p1p0,ctrl_solver)


# solve initial elliptic equation
ierr=1
dmk_solver.syncronize(mkeqs,tdpot,ierr)


# solve initial elliptic equation
tdens_old = Function(p1p0.tdens_fem_space)
time = 0
for i in range(1000):
    tdens_old.vector()[:] = tdpot.tdens.vector()[:]
    time += dmk_solver.ctrl.deltat
    dmk_solver.iterate(mkeqs,tdpot,ierr)
    diff=project(tdens_old-tdpot.tdens,p1p0.tdens_fem_space)
    variation = norm(diff,'l2')
    print(min(tdpot.tdens.vector()),max(tdpot.tdens.vector()))
    print('var tdens = {:.1e}'.format(variation))

    print(variation,variation < 1e-4)
    if (variation < 1e-4):
        break
    

# Save exact gradeint in xdmf format
name_file_out=("runs/ndiv"+str(ndiv)+"tdenspot.xdmf")
file_out=XDMFFile(MPI.comm_world, name_file_out)
file_out.parameters.update(
    {
        "functions_share_mesh": True,
        "rewrite_function_mesh": False
    })
file_out.write(tdpot.tdens, i)
file_out.write(tdpot.pot, i)
f_p0=project(forcing, p1p0.tdens_fem_space)
file_out.write(f_p0, i)
