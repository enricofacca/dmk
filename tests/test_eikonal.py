import sys
import os
from copy import deepcopy as cp

import numpy as np

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
import mshr



# create grid of square [0,1]x[0,1] and hole inside
ndiv = 32
inner_radius = 0.1
domain = mshr.Rectangle(Point(0,0), Point(1,1)) - mshr.Circle(Point(0.5,0.5), inner_radius)
mesh = mshr.generate_mesh(domain, ndiv)


# set expression for forcing term and boundary condition at inner boundary
forcing=Constant(1.0)
tol=1e-1
def inner_boundary(x, on_boundary):
    if on_boundary:
        if ( ((x[0]-1/2)*(x[0]-1/2)+(x[1]-1/2)*(x[1]-1/2)) < inner_radius*inner_radius + tol ):
            return True
        else:
            return False
    else:
        return False

class Distance(UserExpression):
    """
    Define distance from 
    """
    def eval(self, value, x):
        """
        Define Forcing as F(|x,y|)
        """
        value[0]= x[0];#100*((x[0]-1/2)*(x[0]-1/2)+(x[1]-1/2)*(x[1]-1/2))

        
distance_from_source = Expression("sqrt((x[0]-0.5)*(x[0]-0.5)+(x[1]-0.5)*(x[1]-0.5))", degree=6)

print(distance_from_source([0.6,0.5]))
#distance_from_source = Distance()
bc=[inner_boundary, distance_from_source]

# Define problem disctetization
p1p0 = SpaceDiscretization(mesh)

# Create a class descibing problem inputs given
# a Space Discretization and fill it
mkeqs = PLaplacianProblem(p1p0)
mkeqs.set(forcing,[bc])

# Define problem solution
tdpot = TdensPotential(p1p0)


# init iterative solver
ctrl_solver = DmkControls(time_discretization_method='explicit_tdens')
dmk_solver = DmkSolver(p1p0,ctrl_solver)


# solve initial elliptic equation
ierr=1
dmk_solver.syncronize(mkeqs,tdpot,ierr)
i=0
#"""
# solve initial elliptic equation
tdens_old = Function(p1p0.tdens_fem_space)
time = 0
for i in range(100):
    tdens_old.vector()[:] = tdpot.tdens.vector()[:]
    time += dmk_solver.ctrl.deltat
    dmk_solver.iterate(mkeqs,tdpot,ierr)
    diff=project(tdens_old-tdpot.tdens,p1p0.tdens_fem_space)
    variation = norm(diff,'l2')

    print('var tdens = {:.1e}'.format(variation))

    diff=project(distance_from_source-tdpot.pot,p1p0.pot_fem_space)
    error = norm(diff,'l2')
    print('Error distance', error)
    
    if (variation < 1e-3):
        break
#""" 

# Save exact gradeint in xdmf format
name_file_out=("runs/ndiv"+str(ndiv)+"tdenspot.xdmf")
print(name_file_out)
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
