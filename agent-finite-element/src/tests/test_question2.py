r"""
Test module for Question 2.
"""

import os.path
import site

for relpath in ['../Special', '../Common', '../..']:
    site.addsitedir(
        os.path.realpath(os.path.dirname(__file__) + relpath)
    )
    site.addsitedir(
        os.path.realpath(os.path.dirname(__file__) + '/' + relpath)
    )

import unittest

import numpy
import sympy

import BigGMesh
import Vector
import question2

from FiniteElement import *

# Number of nodes for the rectangle from Finite Elements Lab 2.
NODE_COUNT = 15

def generate_2d_grid(Nx):
    r"""
    From Finite Elements Lab 2.
    """
    Nnodes = Nx+1
    x = numpy.linspace(0, 1, Nnodes)
    y = numpy.linspace(0, 1, Nnodes)
    X, Y = numpy.meshgrid(x,y)
    nodes = numpy.zeros((Nnodes**2,2))
    nodes[:,0] = X.ravel()
    nodes[:,1] = Y.ravel()
    ID = numpy.zeros(len(nodes), dtype=numpy.int)
    n_eq = 0
    for nID in range(len(nodes)):
        if nID % Nnodes == Nx:
            ID[nID] = -1
        else:
            ID[nID] = n_eq
            n_eq += 1
    IEN = numpy.zeros((2*Nx**2,3), dtype=numpy.int)
    for i in range(Nx):
        for j in range(Nx):
            IEN[2*i+2*j*Nx  , :] = i+j*Nnodes, i+1+j*Nnodes, i+(j+1)*Nnodes
            IEN[2*i+1+2*j*Nx, :] = i+1+j*Nnodes, i+1+(j+1)*Nnodes, i+(j+1)*Nnodes
    return nodes, IEN, ID

class test_question2(unittest.TestCase):
    def test_simpleHeatSource(self):
        r"""
        Uses a simple heat source function from Finite Elements Lab 2:

        .. math::
            f(x, y) = 1
        """
        matNodePositions, matTriangles, dummy = generate_2d_grid(NODE_COUNT)
        mesh = BigGMesh._meshFromNodePosIen(
            matNodePositions,
            matTriangles,
            lambda pos: nearlyEqual(pos[0], 1.0)
        )

        matStiffness = question2.globalStiffnessMatrix(mesh)

        x, y = sympy.symbols('x, y')
        matInput = question2.globalInputMatrix(mesh, SympyFn((x, y), 1, numpy))

        # Calculate temperatures.
        temperatures = numpy.array(numpy.linalg.solve(matStiffness, matInput)).flatten()

        # Associate each node, including Dirichlet nodes, with temperature.

        nodeToTemp = AssocList(lambda a, b: a is b)

        for node in mesh.nodes():
            if (isinstance(node, NonDirichletNode)):
                nodeToTemp.add(node, temperatures[mesh.getNonDirichletNodeIndex(node)])
            else:
                # Dirichlet boundary condition.
                nodeToTemp.add(node, 0.0)

        # Plot temperatures.
        print('The 2D plot should be similar to the tripcolor plot in Finite Elements Lab 2.')
        question2.plotTemperatures(mesh, nodeToTemp, 0.0667)

if (__name__ == '__main__'):
    unittest.main()
