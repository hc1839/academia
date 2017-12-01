r"""
Node indices and triangles for Question 2.
"""

import numpy
import sympy

import Geometry
import Vector

from FiniteElement import *
from FiniteElementTriangle import *
from Float import nearlyEqual
from List import AssocList

def _find_node_index_of_location(nodes, location):
    """
    Given all the nodes and a location (that should be the location of *a*
    node), return the index of that node.

    nodes : array of float
        (Nnodes, 2) array containing the x, y coordinates of the nodes
    location : array of float
        (2,) array containing the x, y coordinates of location

    :return:
        Index of the node.
    """
    dist_to_location = numpy.linalg.norm(nodes - location, axis=1)
    return numpy.argmin(dist_to_location)


def _generate_g_grid(side_length):
    """
    Generate a 2d triangulation of the letter G. All triangles have the same
    size (right triangles, short length side_length)

    side_length : float
        The length of each triangle. Should be 1/N for some integer N

    :return:
        nodes : array of float
            (Nnodes, 2) array containing the x, y coordinates of the nodes
        IEN : array of int
            (Nelements, 3) array linking element number to node number
        ID : array of int
            (Nnodes,) array linking node number to equation number; value is -1
            if node should not appear in global arrays.
    """
    x = numpy.arange(0, 4+0.5*side_length, side_length)
    y = numpy.arange(0, 5+0.5*side_length, side_length)
    X, Y = numpy.meshgrid(x,y)
    potential_nodes = numpy.zeros((X.size,2))
    potential_nodes[:,0] = X.ravel()
    potential_nodes[:,1] = Y.ravel()
    xp = potential_nodes[:,0]
    yp = potential_nodes[:,1]
    nodes_mask = numpy.logical_or(numpy.logical_and(xp>=2,numpy.logical_and(yp>=2,yp<=3)),
                                  numpy.logical_or(numpy.logical_and(xp>=3,yp<=3),
                                                   numpy.logical_or(xp<=1,
                                                                    numpy.logical_or(yp<=1, yp>=4))))
    nodes = potential_nodes[nodes_mask, :]
    ID = numpy.zeros(len(nodes), dtype=numpy.int)
    n_eq = 0
    for nID in range(len(nodes)):
        if numpy.allclose(nodes[nID,0], 4):
            ID[nID] = -1
        else:
            ID[nID] = n_eq
            n_eq += 1
    inv_side_length = int(1 / side_length)
    Nelements_per_block = inv_side_length**2
    Nelements = 2 * 14 * Nelements_per_block
    IEN = numpy.zeros((Nelements,3), dtype=numpy.int)
    block_corners = [[0,0], [1,0], [2,0], [3,0],
                     [0,1],               [3,1],
                     [0,2],        [2,2], [3,2],
                     [0,3],
                     [0,4], [1,4], [2,4], [3,4]]
    current_element = 0
    for block in block_corners:
        for i in range(inv_side_length):
            for j in range(inv_side_length):
                node_locations = numpy.zeros((4,2))
                for a in range(2):
                    for b in range(2):
                        node_locations[a+2*b,0] = block[0] + (i+a)*side_length
                        node_locations[a+2*b,1] = block[1] + (j+b)*side_length
                index_lo_l = _find_node_index_of_location(nodes, node_locations[0,:])
                index_lo_r = _find_node_index_of_location(nodes, node_locations[1,:])
                index_hi_l = _find_node_index_of_location(nodes, node_locations[2,:])
                index_hi_r = _find_node_index_of_location(nodes, node_locations[3,:])
                IEN[current_element, :] = [index_lo_l, index_lo_r, index_hi_l]
                current_element += 1
                IEN[current_element, :] = [index_lo_r, index_hi_r, index_hi_l]
                current_element += 1
    return nodes, IEN, ID

def _meshFromNodePosIen(nodes, ien, isDirichletNode):
    r"""
    Constructs a mesh using right triangles.

    :param numpy.ndarray nodes:
        Matrix of node positions as row vectors.

    :param numpy.ndarray ien:
        IEN matrix of global node indices, where each row vector represents a
        triangle, and column vectors represent the local node indices in order.

    :param callable isDirichletNode:
        Unary function, where first parameter is node position as a row vector.
        It returns ``True`` if the position corresponds to a Dirichlet node;
        ``False`` if otherwise.

    :return:
        Mesh with isosceles right triangles as elements.

    :rtype: :class:`FiniteElementTriangle.TriangleMesh`

    :exception ValueError:
        Global indices are not unique.

    :exception ValueError:
        Triangle is neither a lower nor an upper.
    """
    matNodePositions = numpy.matrix(nodes)
    matTriangles = numpy.matrix(ien).astype(int)

    xi, eta = sympy.symbols('xi, eta')
    shapeFns = [
        SympyFn((xi, eta), 1 - xi - eta, numpy),
        SympyFn((xi, eta), xi, numpy),
        SympyFn((xi, eta), eta, numpy)
    ]

    scaffold = MeshScaffold(TriangleMesh)

    # Mapping from position vector to global index.
    globalIndices = AssocList(lambda a, b: Vector.nearlyEqualVectors(a, b))

    # Construct triangles.
    for triangleNodeIdxs in Vector.getRowVectors(matTriangles):
        nodes = []

        # Construct nodes.
        for nodePosition, globalIndex in zip(Vector.getRowVectors(matNodePositions[triangleNodeIdxs, :]), triangleNodeIdxs):
            # Determine the proper node type, and create it.
            if (isDirichletNode(nodePosition)):
                nodes.append(NodePlaceholder(DirichletNode, nodePosition.astype(float)))
            else:
                nodes.append(NodePlaceholder(NonDirichletNode, nodePosition.astype(float)))

            if (globalIndices.isMember(nodePosition)):
                if (globalIndices[nodePosition] != globalIndex):
                    raise ValueError('Global indices are not unique.')
            else:
                globalIndices.add(nodePosition, globalIndex)

        # Determine whether the right triangle is lower or upper, and create
        # it.
        if (Geometry.isLowerRightTriangle([node.position() for node in nodes])):
            LowerRightTriangle(scaffold, nodes, shapeFns)
        elif (Geometry.isUpperRightTriangle([node.position() for node in nodes])):
            UpperRightTriangle(scaffold, nodes, shapeFns)
        else:
            raise ValueError('Triangle is neither a lower nor an upper.')

    return scaffold.build(lambda mesh, nodes, newNode: globalIndices[newNode.position()])

def bigGMesh(rightTriangleLegLength):
    r"""
    Constructs the mesh for the Big G from Question 2.

    :param float rightTriangleLegLength:
        Leg length of each isosceles right triangle.

    :return:
        Mesh with isosceles right triangles as elements. Nodes with
        x-coordinate equal to ``4`` are Dirichlet nodes.

    :rtype: :class:`FiniteElementTriangle.TriangleMesh`
    """
    rightTriangleLegLength = float(rightTriangleLegLength)

    matNodePositions, matTriangles, dummy = _generate_g_grid(rightTriangleLegLength)

    return _meshFromNodePosIen(
        matNodePositions,
        matTriangles,
        lambda pos: nearlyEqual(pos[0], 4.0)
    )
