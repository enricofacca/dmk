import sys
import os
from copy import deepcopy as cp

sys.path.append('../src/dmk')
from dmk import SpaceDiscretization
from dmk import TdensPotential
from dmk import DmkControls
from dmk_minimal_surface import MinimalSurfaceProblem
from dmk_minimal_surface import MinimalSurfaceSolver

from dolfin import UserExpression
from dolfin import UnitSquareMesh
from dolfin import *

from dolfin import norm


tol = 1E-14


def boundary_one(x, on_boundary):
    if on_boundary:
        if near(x[0], 0, tol) and (x[1]>0.25) and (x[1]<0.75):
            return True
        elif near(x[0], 1, tol) and (x[1]>0.75):
            return True
        else:
            return False
    else:
        return False
pot_at_boundary_one = Constant(1.0)
    
def boundary_zero(x, on_boundary):
    if on_boundary:
        return not boundary_one(x,on_boundary)
    else:
        return False
pot_at_boundary_zero = Constant(0.0)

bcs = [
    [boundary_one, pot_at_boundary_one],
    [boundary_zero, pot_at_boundary_zero]
]

def boundary_x(x, on_boundary):
    if on_boundary:
        if near(x[0], 0, tol) or near(x[0], 1, tol):
            return True
        else:
            return False
    else:
        return False
pot_at_x =  Expression("abs(x[1]-0.5)", degree=2)

def boundary_y(x, on_boundary):
    if on_boundary:
        if near(x[1], 0, tol) or near(x[1], 1, tol):
            return True
        else:
            return False
    else:
        return False
pot_at_y =  Expression("abs(x[0]-0.5)", degree=2)

bcs = [
    [boundary_x, pot_at_x],
    [boundary_y, pot_at_y]
]

#"""
def boundary_zero(x, on_boundary):
    return on_boundary
pot = Expression('abs(x[0]-0.5)*(1-x[1]*(x[1]-1))+abs(x[1]-0.5)*(1-x[0]*(x[0]-1))', degree=2)

bcs = [
    [boundary_zero, pot]
]
#"""

# create uniform grid of square [0,1]x[0,1]
ndiv = 64
mesh = UnitSquareMesh(ndiv, ndiv)

# set expression for forcing term
forcing=Constant(0.0)


# Define problem disctetization
crp0 = SpaceDiscretization(mesh)

# Create a class descibing problem inputs given
# a Space Discretization and fill it
minimal_surface_square = MinimalSurfaceProblem(crp0)
minimal_surface_square.set(forcing,bcs)

# Define problem solution
tdpot = TdensPotential(crp0)

# init iterative solver
ctrl_solver = DmkControls(time_discretization_method='explicit_tdens',deltat=0.05)
minimal_surface_solver = MinimalSurfaceSolver(crp0,ctrl_solver)


# solve initial elliptic equation
ierr=1
minimal_surface_solver.syncronize(minimal_surface_square,tdpot,ierr)


# solve initial elliptic equation
tdens_old = Function(crp0.tdens_fem_space)
time = 0
for i in range(1000):
    tdens_old.vector()[:] = tdpot.tdens.vector()[:]
    time += minimal_surface_solver.ctrl.deltat
    minimal_surface_solver.iterate(minimal_surface_square,tdpot,ierr)
    diff=project(tdens_old-tdpot.tdens,crp0.tdens_fem_space)
    variation = norm(diff,'l2')
    print(min(tdpot.tdens.vector()),max(tdpot.tdens.vector()))
    print('var tdens = {:.1e}'.format(variation))

    if (variation < 1e-3):
        break

# Save exact gradeint in xdmf format
name_file_out=("runs/minimal_surface/ndiv"+str(ndiv)+"tdenspot.xdmf")
file_out=XDMFFile(MPI.comm_world, name_file_out)
file_out.parameters.update(
    {
        "functions_share_mesh": True,
        "rewrite_function_mesh": False
    })
file_out.write(tdpot.tdens, i)
file_out.write(tdpot.pot, i)
f_p0=project(forcing, crp0.tdens_fem_space)
file_out.write(f_p0, i)
