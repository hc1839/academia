r"""
Geometry-related functions.

Vectors are represented as ``numpy.ndarray``, and matrices are represented as
``numpy.matrix``.
"""

import numpy

import Vector

from Float import nearlyEqual
from Vector import isVector

def isTriangle(positions):
    r"""
    Determines whether a sequence of positions represents the vertices of a
    triangle.

    :param positions:
        Sequence of position vectors.

    :return:
        ``True`` if the positions represent the vertices of a triangle;
        ``False`` if otherwise.

    :rtype: bool
    """
    positions = list(positions)

    if (len(list(filter(lambda elt: not(isVector(elt)), positions))) > 0):
        raise TypeError()

    if (len(positions) != 3):
        return False

    # Two of the edges.
    twoEdges = [
        positions[1] - positions[0],
        positions[2] - positions[0]
    ]

    # Magnitudes of the two edges.
    twoEdgesMagnitudes = [
        Vector.magnitude(vec) for vec in twoEdges
    ]

    if (nearlyEqual(twoEdgesMagnitudes[0], 0.0) or nearlyEqual(twoEdgesMagnitudes[1], 0.0)):
        return False

    # Determine whether the positions are collinear.
    if (nearlyEqual(numpy.arccos(numpy.vdot(twoEdges[0], twoEdges[1]) / twoEdgesMagnitudes[0] / twoEdgesMagnitudes[1]), 0.0)):
        return False

    return True

def isRightTriangle(positions):
    r"""
    Determines whether a sequence of positions represents the vertices of a
    right triangle.

    :param positions:
        Sequence of position vectors.

    :return:
        ``True`` if the positions represent the vertices of a right triangle;
        ``False`` if otherwise.

    :rtype: bool
    """
    positions = list(positions)

    if (not(isTriangle(positions))):
        return False

    for position in positions:
        otherPositions = list(filter(lambda elt: not(Vector.nearlyEqualVectors(elt, position)), positions))

        # Two of the edges.
        twoEdges = [
            otherPosition - position
            for otherPosition in otherPositions
        ]

        # Magnitudes of the two edges.
        twoEdgesMagnitudes = [
            Vector.magnitude(vec) for vec in twoEdges
        ]

        if (nearlyEqual(numpy.arccos(numpy.vdot(twoEdges[0], twoEdges[1]) / twoEdgesMagnitudes[0] / twoEdgesMagnitudes[1]), numpy.pi / 2.0)):
            return True

    return False

def rightAnglePositionOfTriangle(positions):
    r"""
    Position of the right angle of a triangle.

    :param iterable positions:
        Sequence of position vectors for the vertices.

    :return:
        Position of the right angle.

    :rtype: ``numpy.ndarray``

    :exception ValueError:
        Not a right triangle.
    """
    positions = list(positions)

    if (not(isRightTriangle(positions))):
        raise ValueError('Not a right triangle.')

    for position in positions:
        otherPositions = list(filter(lambda elt: not(Vector.nearlyEqualVectors(elt, position)), positions))

        # Two of the edges.
        twoEdges = [
            otherPosition - position
            for otherPosition in otherPositions
        ]

        # Magnitudes of the two edges.
        twoEdgesMagnitudes = [
            Vector.magnitude(vec) for vec in twoEdges
        ]

        if (nearlyEqual(numpy.arccos(numpy.vdot(twoEdges[0], twoEdges[1]) / twoEdgesMagnitudes[0] / twoEdgesMagnitudes[1]), numpy.pi / 2.0)):
            return position

    raise RuntimeError('Position of the right angle cannot be found.')

def isLowerRightTriangle(positions):
    r"""
    Determines whether a sequence of positions represents the vertices of a
    lower right triangle.

    :param positions:
        Sequence of position vectors.

    :return:
        ``True`` if the positions represent the vertices of a lower right
        triangle; ``False`` if otherwise.

    :rtype: bool
    """
    positions = list(positions)

    if (not(isRightTriangle(positions))):
        return False

    # Sort positions by x- and y-coordinates, separately.
    sortedPositionsByXCoord = sorted(positions, key = lambda elt: elt[0])
    sortedPositionsByYCoord = sorted(positions, key = lambda elt: elt[1])

    rightAnglePosition = rightAnglePositionOfTriangle(positions)

    # Supposed bottom and left edges.
    bottomEdge = sortedPositionsByXCoord[2] - rightAnglePosition
    leftEdge = sortedPositionsByYCoord[2] - rightAnglePosition

    if (not(nearlyEqual(bottomEdge[1], 0.0) and bottomEdge[0] > 0.0)):
        return False

    if (not(nearlyEqual(leftEdge[0], 0.0) and leftEdge[1] > 0.0)):
        return False

    return True

def isUpperRightTriangle(positions):
    r"""
    Determines whether a sequence of positions represents the vertices of a
    upper right triangle.

    :param positions:
        Sequence of position vectors.

    :return:
        ``True`` if the positions represent the vertices of a upper right
        triangle; ``False`` if otherwise.

    :rtype: bool
    """
    positions = list(positions)

    if (not(isRightTriangle(positions))):
        return False

    # Sort nodes by x- and y-coordinates, separately.
    sortedPositionsByXCoord = sorted(positions, key = lambda elt: elt[0])
    sortedPositionsByYCoord = sorted(positions, key = lambda elt: elt[1])

    rightAnglePosition = rightAnglePositionOfTriangle(positions)

    # Supposed right and top edges.
    rightEdge = sortedPositionsByYCoord[0] - rightAnglePosition
    topEdge = sortedPositionsByXCoord[0] - rightAnglePosition

    if (not(nearlyEqual(rightEdge[0], 0.0) and rightEdge[1] < 0.0)):
        return False

    if (not(nearlyEqual(topEdge[1], 0.0) and topEdge[0] < 0.0)):
        return False

    return True
