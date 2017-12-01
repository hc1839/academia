#!/usr/bin/env python3

import os.path
import site

for relpath in ['src/Special', 'src/Common']:
    site.addsitedir(
        os.path.realpath(os.path.dirname(__file__) + relpath)
    )
    site.addsitedir(
        os.path.realpath(os.path.dirname(__file__) + '/' + relpath)
    )

import itertools
import numpy
import numpy.matlib
import operator
import scipy.integrate
import sympy
import warnings

import Vector

from BigGMesh import bigGMesh
from FiniteElement import *
from Float import nearlyEqual
from List import AssocList
from PrintedLine import printWithCr
from SympyFn import SympyFn
from matplotlib import pyplot
from mpl_toolkits.mplot3d import Axes3D

# Leg lengths of the triangles.
TRIANGLE_LEG_LENGTHS = [1.0 / 2.0, 1.0 / 16.0]

# String prepended to the operation name when printing to standard output.
CURRENT_OPERATION_INDICATOR = 'Current operation: '

def elementStiffnessMatrixTerm(element, localIndex1, localIndex2):
    r"""
    Calculates the stiffness matrix term for two nodes in an element.

    It uses the unit triangle as parent element.

    :param FiniteElement.Element element:
        Element.

    :param sequence localIndex1:
        A local index.

    :param sequence localIndex2:
        Another local index.

    :return:
        Stiffness matrix term for the nodes at the two local indices in an
        element. Swapping ``localIndex1`` and ``localIndex2`` gives the same
        result.

    :rtype: float
    """
    if (not(isinstance(element, Element))):
        raise TypeError()

    localIndex1 = int(localIndex1)
    localIndex2 = int(localIndex2)

    # Construct matrix of gradients of the two shape functions as column
    # vectors in Sympy expressions.
    (xi, eta), matShapeFnGradientsParent = Vector.sympyOfFnMatrix(numpy.hstack([
        Vector.asColumn(
            numpy.array(
                Vector.gradient(element.getShapeFn(localIndex)).components()
            )
        )
        for localIndex in [localIndex1, localIndex2]
    ]))

    # Jacobian for the chain rule used in coordinate transformation.
    elementJacobian = Vector.sympyOfFnMatrix(
        Vector.jacobian(element.coordTransformation(numpy), asColumnVectors = True)
    )[1]

    starJacobian = elementJacobian.inv()
    matShapeFnGradientsMesh = starJacobian.multiply(matShapeFnGradientsParent)
    stiffnessIntegrand = SympyFn((xi, eta), matShapeFnGradientsMesh[:, 0].T.multiply(matShapeFnGradientsMesh[:, 1])[0, 0], numpy)

    # Specify the relationship between xi and eta, which is the unit triangle,
    # for the double integration.
    xiFn = SympyFn((eta,), 1 - eta, numpy)

    # Integrate with respect to parent element coordinates.
    return scipy.integrate.dblquad(stiffnessIntegrand, 0.0, 1.0, lambda ordinate: 0.0, xiFn)[0]

def globalStiffnessMatrix(mesh):
    r"""
    Calculates the global stiffness matrix for a mesh.

    :param FiniteElement.Mesh mesh:
        Mesh.

    :return:
        Global stiffness matrix.

    :rtype: ``numpy.matrix``
    """
    if (not(isinstance(mesh, Mesh))):
        raise TypeError()

    cntNonDirichletNode = len(list(filter(lambda node: isinstance(node, NonDirichletNode), mesh.nodes())))

    # Stiffness Matrix.
    matStiffness = numpy.matlib.zeros((cntNonDirichletNode, cntNonDirichletNode), dtype = float)

    # Calculate stiffness matrix for each element.
    for element in mesh.elements():
        # Local index pair to stiffness for the two nodes.
        localIndexPairToStiffness = AssocList(lambda a, b: a == b or a == (b[1], b[0]))

        localIndexPairs = list(itertools.product([localIndex for localIndex in range(3)], repeat = 2))

        # Remove pairs that contain local indices of Dirichlet nodes.
        for i in range(len(localIndexPairs) - 1, -1, -1):
            localIndexPair = localIndexPairs[i]
            nodes = [element.getNode(localIndex) for localIndex in localIndexPair]

            if (len(list(filter(lambda elt: isinstance(elt, DirichletNode), nodes))) > 0):
                del localIndexPairs[i]

        # Calculate stiffness, if necessary, for all possible pairs of local
        # indices for non-Dirichlet nodes.
        for localIndexPair in localIndexPairs:
            if (not(localIndexPairToStiffness.isMember(localIndexPair))):
                # Add the stiffness for the two nodes.
                localIndexPairToStiffness.add(
                    localIndexPair,
                    elementStiffnessMatrixTerm(element, localIndexPair[0], localIndexPair[1])
                )

            nonDirichletNodeIndices = tuple(
                mesh.getNonDirichletNodeIndex(element.getNode(localIndex))
                for localIndex in localIndexPair
            )

            # Add the element stiffness to the matrix of non-Dirichlet node
            # stiffnesses.
            matStiffness[nonDirichletNodeIndices] += localIndexPairToStiffness[localIndexPair]

    return matStiffness

def elementInputMatrixTerm(heatSourceFn, element, localIndex):
    r"""
    Calculates the input matrix (force vector) term for a node in an element.

    It uses the unit triangle as parent element.

    :param SympyFn.SympyFn heatSourceFn:
        Heat source function.

    :param FiniteElement.Element element:
        Element.

    :param int localIndex:
        Local index of the node.

    :return:
        Input matrix term for the node with the local index in an element.

    :rtype: float
    """
    # Declarations.
    localShapeFn = None
    coordTransformation = None

    def inputIntegrand(abscissa, ordinate):
        r"""
        Abscissa and ordinate are in parent element coordinates.
        """
        return localShapeFn(abscissa, ordinate) * heatSourceFn(*coordTransformation(abscissa, ordinate))

    if (not(isinstance(heatSourceFn, SympyFn))):
        raise TypeError()

    if (not(isinstance(element, Element))):
        raise TypeError()

    localIndex = int(localIndex)

    # Shape function at the given local index.
    localShapeFn = element.getShapeFn(localIndex)

    # From parent element to mesh element.
    coordTransformation = element.coordTransformation(numpy)

    # Parameters of a shape function in parent element coordinates.
    xi, eta = localShapeFn.parameters()

    # Specify the relationship between xi and eta, which is the unit triangle,
    # for the double integration.
    xiFn = SympyFn((eta,), 1 - eta, numpy)

    # Integrate with respect to parent element coordinates.
    return scipy.integrate.dblquad(inputIntegrand, 0.0, 1.0, lambda ordinate: 0.0, xiFn)[0]

def globalInputMatrix(mesh, heatSourceFn):
    r"""
    Calculates the global input matrix (force vector) for a mesh.

    :param FiniteElement.Mesh mesh:
        Mesh.

    :param SympyFn.SympyFn heatSourceFn:
        Heat source function.

    :return:
        Global input matrix.

    :rtype: ``numpy.matrix``
    """
    if (not(isinstance(mesh, Mesh))):
        raise TypeError()

    if (not(isinstance(heatSourceFn, SympyFn))):
        raise TypeError()

    cntNonDirichletNode = len(list(filter(lambda node: isinstance(node, NonDirichletNode), mesh.nodes())))

    # Input matrix (force vector).
    matInput = numpy.matlib.zeros((cntNonDirichletNode, 1), dtype = float)

    # Calculate input matrix for each element.
    for element in mesh.elements():
        # Local indices of non-Dirichlet nodes.
        localIndices = [
            localIndex
            for localIndex in range(3)
            if (isinstance(element.getNode(localIndex), NonDirichletNode))
        ]

        for localIndex in localIndices:
            node = element.getNode(localIndex)
            matInput[mesh.getNonDirichletNodeIndex(node), 0] += elementInputMatrixTerm(heatSourceFn, element, localIndex)

    return matInput

def plotTemperatures(mesh, nodeToTemp, triangleLegLength):
    r"""
    Plots the temperatures at the nodes of a mesh.

    :param FiniteElement.Mesh mesh:
        Mesh.

    :param List.AssocList nodeToTemp:
        Association list that maps from :class:`FiniteElement.Node` object to
        temperature at the node.

    :param float triangleLegLength:
        Leg length of the right triangle.

    :return:
        ``None``
    """
    if (not(isinstance(mesh, Mesh))):
        raise TypeError()

    if (not(isinstance(nodeToTemp, AssocList))):
        raise TypeError()

    # Plot the temperatures in 2D.
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')

        # Nodes ordered by global index.
        orderedNodes = list(sorted(mesh.nodes(), key = lambda node: node.globalIndex()))

        # Row vectors are positions.
        matPositions = numpy.array([
            node.position() for node in orderedNodes
        ])

        # Matrix of node indices. Row vector is three node indices
        # representing a triangle.
        matTriangles = numpy.array([
            [node.globalIndex() for node in element.nodes()]
            for element in mesh.elements()
        ])

        # Temperatures ordered by global index.
        orderedTemperatures = [
            nodeToTemp[node] for node in orderedNodes
        ]

        fig = pyplot.figure()

        ax = fig.add_subplot(1, 1, 1)
        ax.axis('equal')
        ax.set_xlabel(r'$x$')
        ax.set_ylabel(r'$y$');
        ax.set_title('Temperatures with Triangle Leg Length of ' + str(triangleLegLength))
        ax.tripcolor(matPositions[:, 0], matPositions[:, 1], orderedTemperatures, triangles = matTriangles)

        pyplot.show()

    # Plot the temperatures in 3D.
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')

        fig = pyplot.figure()

        ax = fig.add_subplot(1, 1, 1, projection = '3d')
        ax.set_xlabel(r'$x$')
        ax.set_ylabel(r'$y$')
        ax.set_zlabel(r'Temperature')
        ax.set_title('Temperatures with Triangle Leg Length of ' + str(triangleLegLength))
        ax.plot_trisurf(matPositions[:, 0], matPositions[:, 1], orderedTemperatures, triangles = matTriangles)

        pyplot.show()

if (__name__ == '__main__'):

    # Determine the heat equation solution for the different triangle leg lengths.
    for triangleLegLength in TRIANGLE_LEG_LENGTHS:
        print('Determining temperatures for triangle leg length of ' + str(triangleLegLength) + '.')

        printWithCr(CURRENT_OPERATION_INDICATOR + 'Generating mesh...')

        # Generate mesh.
        mesh = bigGMesh(triangleLegLength)

        printWithCr(CURRENT_OPERATION_INDICATOR + 'Calculating stiffness matrix...')

        # Calculate stiffness matrix.
        matStiffness = globalStiffnessMatrix(mesh)

        printWithCr(CURRENT_OPERATION_INDICATOR + 'Calculating force vector...')

        # Heat source function.
        x, y = sympy.symbols('x, y')
        heatSourceFn = SympyFn((x, y), sympy.exp(- (x ** 2 + y ** 2)), numpy)

        # Calculate input matrix (force vector).
        matInput = globalInputMatrix(mesh, heatSourceFn)

        printWithCr(CURRENT_OPERATION_INDICATOR + 'Calculating temperatures...')

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

        print()
        print('Complete.')

        # Plot temperatures.
        plotTemperatures(mesh, nodeToTemp, triangleLegLength)
