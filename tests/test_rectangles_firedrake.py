import sys
import os
from copy import deepcopy as cp

sys.path.append('../src/dmk')
from dmk_firedrake import SpaceDiscretization
from dmk_firedrake import TdensPotential
from dmk_firedrake import DmkSolver
from dmk_firedrake import DmkControls
from dmk_firedrake import PLaplacianProblem

from ufl import *
from ufl.classes import Expr
from firedrake import UnitSquareMesh

from firedrake import norm
from firedrake import Function
from firedrake import interpolate
# for writing to file
from firedrake import File


class Rect(Expr):
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



class OptimalTdensRect(Expr):
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
ndiv = int(sys.argv[1])
mesh = UnitSquareMesh(ndiv, ndiv)

# Define problem disctetization
p1p0 = SpaceDiscretization(mesh)

# set expression for forcing term
#forcing = Rect()
x, y = SpatialCoordinate(mesh)
source  = conditional( And(abs(x-0.25) < 0.125, abs(y-0.5)<0.25), 2.0, 0.0) 
sink    = conditional( And(abs(x-0.75) < 0.125, abs(y-0.5)<0.25), 2.0, 0.0) 
forcing = source - sink


f = Function(p1p0.tdens_fem_space)
f.interpolate(forcing) 


optimal_tdens = OptimalTdensRect()




# Create a class descibing problem inputs given
# a Space Discretization and fill it
mkeqs = PLaplacianProblem(p1p0)
mkeqs.set(f)

# Define problem solution
tdpot = TdensPotential(p1p0)

# init iterative solver
ctrl_solver = DmkControls(time_discretization_method='explicit_tdens')
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
    diff=interpolate(tdens_old-tdpot.tdens,p1p0.tdens_fem_space)
    variation = norm(diff,'l2')
    print(min(tdpot.tdens.vector()),max(tdpot.tdens.vector()))
    print('var tdens = {:.1e}'.format(variation))

    if (variation < 1e-2):
        break

out_file_name=("runs_firedrake/ndiv"+str(ndiv)+"tdenspot.pvd")
out_file = File(out_file_name)
out_file.write(f, time=0)
#out_file.write(tdpot.pot,time=0)
