r"""
Finite element method using triangles.
"""

import abc
import numpy
import weakref

import Geometry
import Vector

from FiniteElement import *
from Float import nearlyEqual
from List import AssocList

ABC = abc.ABCMeta('ABC', (object,), {'__slots__': ()})

class Triangle(Element2d):
    r"""
    Triangular element.
    """

    def __init__(self, mesh, nodes, shapeFns):
        r"""
        :param Mesh triangleMesh:
            See :func:`FiniteElement.Element.__init__`.

        :param Node nodes:
            Nodes representing the vertices of the triangle. See
            :func:`FiniteElement.Element.__init__`.

        :param shapeFns:
            See :func:`FiniteElement.Element.__init__`.

        :exception ValueError:
            Nodes do not represent a triangle.
        """
        Element.__init__(self, mesh, nodes, shapeFns)
        nodes = list(Element.nodes(self))

        if (not(isinstance(mesh, Mesh))):
            raise TypeError()

        if (not(Geometry.isTriangle([node.position() for node in nodes]))):
            raise ValueError('Nodes do not represent a triangle.')

class RightTriangle(Triangle, ABC):
    r"""
    Element that is a right triangle.
    """

    def __init__(self, triangleMesh, nodes, shapeFns):
        r"""
        :param TriangleMesh triangleMesh:
            See :func:`Triangle.__init__`.

        :param Node nodes:
            See :func:`Triangle.__init__`.

        :param shapeFns:
            See :func:`Triangle.__init__`.

        :exception ValueError:
            Nodes do not specify a right triangle.
        """
        Triangle.__init__(self, triangleMesh, nodes, shapeFns)
        nodes = list(Triangle.nodes(self))

        if (not(Geometry.isRightTriangle([node.position() for node in nodes]))):
            raise ValueError('Nodes do not specify a right triangle.')

    def rightAngleNode(self):
        r"""
        Node at the right angle of the triangle.

        :return:
            Right angle node.

        :rtype: :class:`Node`

        :exception RuntimeError:
            Right angle node cannot be found.
        """
        nodes = list(self.nodes())
        positions = [node.position() for node in nodes]

        positionToNode = AssocList(
            lambda a, b: Vector.nearlyEqualVectors(a, b),
            zip(positions, nodes)
        )

        try:
            return positionToNode[Geometry.rightAnglePositionOfTriangle(positions)]
        except:
            raise RuntimeError('Right angle node cannot be found.')

class LowerRightTriangle(RightTriangle):
    r"""
    Element that is a lower right triangle.
    """

    def __init__(self, triangleMesh, nodes, shapeFns):
        r"""
        :param TriangleMesh triangleMesh:
            See :func:`RightTriangle.__init__`.

        :param Node nodes:
            See :func:`RightTriangle.__init__`.

        :param shapeFns:
            See :func:`RightTriangle.__init__`.

        :exception ValueError:
            Nodes do not specify a lower right triangle.

        **Data members:**
            ``_nodes``: List of :class:`Node` objects ordered by local index.
        """
        RightTriangle.__init__(self, triangleMesh, nodes, shapeFns)
        nodes = list(RightTriangle.nodes(self))

        if (not(Geometry.isLowerRightTriangle([node.position() for node in nodes]))):
            raise ValueError('Nodes do not specify a lower right triangle.')

        # Sort nodes by x- and y-coordinates, separately.
        sortedNodesByXCoord = sorted(nodes, key = lambda elt: elt.position()[0])
        sortedNodesByYCoord = sorted(nodes, key = lambda elt: elt.position()[1])

        rightAngleNode = RightTriangle.rightAngleNode(self)
        sortedNodesByLocalIndex = [rightAngleNode]

        # Add nodes in order of local indices.
        sortedNodesByLocalIndex.append(sortedNodesByXCoord[2])
        sortedNodesByLocalIndex.append(sortedNodesByYCoord[2])

        self._nodes = sortedNodesByLocalIndex

    def nodes(self):
        r"""
        Overrides :func:`RightTriangle.nodes`.

        :return:
            Nodes representing the vertices of the triangle. To get a node by
            its local index, use :func:`getNode`.

        :rtype:
            sequence of :class:`Node` objects.
        """
        return self._nodes.copy()

    def getNode(self, localIndex):
        r"""
        Overrides :func:`RightTriangle.getNode`.
        """
        localIndex = int(localIndex)

        if (not(localIndex in [0, 1, 2])):
            raise ValueError('Invalid local index.')

        return self.nodes()[localIndex]

class UpperRightTriangle(RightTriangle):
    r"""
    Element that is an upper right triangle.
    """

    def __init__(self, triangleMesh, nodes, shapeFns):
        r"""
        :param TriangleMesh triangleMesh:
            See :func:`RightTriangle.__init__`.

        :param Node nodes:
            See :func:`RightTriangle.__init__`.

        :param shapeFns:
            See :func:`RightTriangle.__init__`.

        :exception ValueError:
            Nodes do not specify an upper right triangle.

        **Data members:**
            ``_nodes``: List of :class:`Node` objects ordered by local index.
        """
        RightTriangle.__init__(self, triangleMesh, nodes, shapeFns)
        nodes = list(RightTriangle.nodes(self))

        if (not(Geometry.isUpperRightTriangle([node.position() for node in nodes]))):
            raise ValueError('Nodes do not specify an upper right triangle.')

        # Sort nodes by x- and y-coordinates, separately.
        sortedNodesByXCoord = sorted(nodes, key = lambda elt: elt.position()[0])
        sortedNodesByYCoord = sorted(nodes, key = lambda elt: elt.position()[1])

        rightAngleNode = RightTriangle.rightAngleNode(self)
        sortedNodesByLocalIndex = []

        # Add nodes in order of local indices.
        sortedNodesByLocalIndex.append(sortedNodesByYCoord[0])
        sortedNodesByLocalIndex.append(rightAngleNode)
        sortedNodesByLocalIndex.append(sortedNodesByXCoord[0])

        self._nodes = sortedNodesByLocalIndex

    def nodes(self):
        r"""
        Overrides :func:`RightTriangle.nodes`.

        :return:
            Nodes representing the vertices of the triangle. To get a node by
            its local index, use :func:`getNode`.

        :rtype:
            sequence of :class:`Node` objects.
        """
        return self._nodes.copy()

    def getNode(self, localIndex):
        r"""
        Overrides :func:`RightTriangle.getNode`.
        """
        localIndex = int(localIndex)

        if (not(localIndex in [0, 1, 2])):
            raise ValueError('Invalid local index.')

        return self.nodes()[localIndex]

class TriangleMesh(Mesh2d):
    r"""
    2D mesh using triangles as elements.
    """

    def __init__(self):
        r"""
        ``[protected]`` Friend of :func:`FiniteElement.MeshScaffold.build`.
        """
        Mesh2d.__init__(self)

    def _addElement(self, triangle):
        r"""
        Overrides :func:`FiniteElement.Mesh2d._addElement`. Friend of
        :func:`FiniteElement.MeshScaffold.build`.

        :param Triangle triangle:
            Triangle element.

        :return:
            See :func:`FiniteElement.Mesh2d._addElement`.
        """
        if (not(isinstance(triangle, Triangle))):
            raise TypeError()

        Mesh2d._addElement(self, triangle)
