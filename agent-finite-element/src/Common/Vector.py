r"""
Miscellaneous vector functions.

Vectors are represented as ``numpy.ndarray``, and matrices are represented as
``numpy.matrix``.
"""

import functools
import numpy
import numpy.matlib
import sympy
import sys

from SympyFn import SympyFn

from Float import nearlyEqual

class VectorFn(object):
    r"""
    Vector function.

    **Data members:**
        ``_components``: Tuple of components. See :func:`components`.
    """

    def __init__(self, components):
        r"""
        :param iterable components:
            Sequence of :class:`SympyFn.Sympy.Fn` objects representing the
            components of the vector.

        :exception ValueError:
            Components do not have the same parameters.
        """
        components = list(components)
        cmptParams = set()

        for cmpt in components:
            if (not(isinstance(cmpt, SympyFn))):
                raise TypeError()

            cmptParams.add(cmpt.parameters())

        if (len(cmptParams) != 0):
            if (len(cmptParams) != 1):
                raise ValueError('Components do not have the same parameters.')

        self._components = components

    def __call__(self, *argv):
        r"""
        Evaluates the vector function by evaluating the components with given
        arguments.

        :param argv:
            Arguments for each component of the vector function.

        :return:
            Tuple of components of the returned vector from the evaluation.

        :rtype: tuple
        """
        return tuple(cmpt(*argv) for cmpt in self.components())

    def __iter__(self):
        r"""
        Iterator of components of the vector.
        """
        for cmpt in self.components():
            yield cmpt

    def __len__(self):
        r"""
        Number of components.

        :return:
            Number of components.

        :rtype: int
        """
        return len(list(self.components()))

    def __getitem__(self, key):
        r"""
        :param int key:
            Zero-based index of the component to get.

        :return:
            Component with ``key`` as the index.

        :rtype: :class:`SympyFn.SympyFn`
        """
        key = int(key)
        return self.components()[key]

    def components(self):
        r"""
        Components of the vector.

        :return:
            Components.

        :rtype: tuple of :class:`SympyFn.SympyFn` objects
        """
        return self._components

def new(*args):
    r"""
    Construct a vector.

    :param args:
        Vector components.

    :return:
        Vector as ``numpy.ndarray``.
    """
    return numpy.array(args)

def isVector(ndarrayObj):
    r"""
    Determines whether a ``numpy.ndarray`` is a vector.

    :param numpy.ndarray ndarrayObj:
        Numpy array object.

    :return:
        ``True`` if ``ndarrayObj`` is a vector; ``False`` if otherwise.

    :rtype: bool
    """
    if (not(isinstance(ndarrayObj, numpy.ndarray))):
        return False

    return len(ndarrayObj.shape) == 1

def asColumn(v):
    r"""
    Converts a vector to a matrix with one column vector.

    :param v:
        Vector.

    :return:
        Matrix with ``v`` as a column vector.
    """
    if (not(isVector(v))):
        raise TypeError()

    return numpy.matrix(v).T

def asRow(v):
    r"""
    Converts a vector to a matrix with one row vector.

    :param v:
        Vector.

    :return:
        Matrix with ``v`` as a row vector.
    """
    if (not(isVector(v))):
        raise TypeError()

    return numpy.matrix(v)

def magnitude(v):
    r"""
    Magnitude of a vector.

    :param v:
        Vector.

    :return:
        Magnitude of ``v``.

    :rtype: float
    """
    if (not(isVector(v))):
        raise TypeError()

    return numpy.sqrt(numpy.vdot(v, v))

def normalized(v):
    r"""
    Normalize a vector.

    :param v:
        Vector to be normalized.

    :return:
        Unit vector in the direction of ``v``.

    :exception ValueError:
        Vector magnitude is zero.
    """
    if (not(isVector(v))):
        raise TypeError()

    vectorMagnitude = magnitude(v)

    if (nearlyEqual(vectorMagnitude, 0.0)):
        raise ValueError('Vector magnitude is zero.')

    return v / vectorMagnitude

def nearlyEqualVectors(v1, v2, epsilon = sys.float_info.min):
    r"""
    Determines whether two vectors are nearly equal using floating-point
    comparison with :func:`Float.nearlyEqual`.

    :param v1:
        Vector.

    :param v2:
        Vector.

    :param epsilon:
        Epsilon to use for :func:`Float.nearlyEqual`.

    :return:
        ``True`` if ``v1`` and ``v2`` are nearly equal according to
        :func:`Float.nearlyEqual`; ``False`` if otherwise.

    :rtype:
        bool

    :exception ValueError:
        Vectors do not have the same dimensionality.
    """
    if (not(isVector(v1) and isVector(v2))):
        raise TypeError()

    if (v1.shape[0] != v2.shape[0]):
        raise ValueError('Vectors do not have the same dimensionality.')

    for elt in zip(v1, v2):
        if (not(nearlyEqual(elt[0], elt[1], epsilon))):
            return False

    return True

def setUnion(a, b, epsilon = sys.float_info.min):
    r"""
    Set union for two sequence of vectors of the same dimension using
    floating-point comparison with :func:`Float.nearlyEqual`.

    :param iterable a:
        Sequence of vectors.

    :param iterable b:
        Sequence of vectors.

    :param float epsilon:
        Epsilon to use for :func:`Float.nearlyEqual`.

    :return:
        Set union of ``a`` and ``b`` as a list.

    :rtype: list

    :exception ValueError:
        Vectors are not of the same dimension.
    """
    a = list(a)
    b = list(b)
    cat = a + b

    for elt in cat:
        if (not(isVector(elt))):
            raise TypeError()

    if (len(frozenset(map(lambda elt: len(elt), cat))) > 1):
        raise ValueError('Vectors are not of the same dimension.')

    listBuilder = []
    for elt in cat:
        if (len(list(filter(lambda xi: nearlyEqualVectors(xi, elt, epsilon), listBuilder))) == 0):
            listBuilder.append(elt)

    return listBuilder

def setDifference(a, b, epsilon = sys.float_info.min):
    r"""
    Set difference for two sequence of vectors of the same dimension using
    floating-point comparison with :func:`Float.nearlyEqual`.

    :param iterable a:
        Sequence of vectors as the minuend.

    :param iterable b:
        Sequence of vectors as the subtrahend.

    :param iterable epsilon:
        Epsilon to use for :func:`Float.nearlyEqual`.

    :return:
        Set difference, ``a`` - ``b``, as a list.

    :rtype: list

    :exception ValueError:
        Vectors are not of the same dimension.
    """
    a = list(a)
    b = list(b)
    cat = a + b

    for elt in cat:
        if (not(isVector(elt))):
            raise TypeError()

    if (len(frozenset(map(lambda elt: len(elt), cat))) > 1):
        raise ValueError('Vectors are not of the same dimension.')

    if (len(a) == 0):
        return []

    listBuilder = a.copy()
    for eltB in b:
        listBuilder = list(filter(lambda eltA: not(nearlyEqualVectors(eltA, eltB, epsilon)), listBuilder))

        if (len(listBuilder) == 0):
            return []

    return list(listBuilder)

def intRound(v):
    r"""
    Rounds the vector components to integers.

    :param v:
        Vector to be rounded.

    :return:
        Vector with components rounded to the nearest integer and are stored as
        integers.
    """
    if (not(isVector(v))):
        raise TypeError()

    return numpy.array(list(map(lambda elt: int(round(elt)), v.tolist())), dtype = int)

def gradient(sympyFn):
    r"""
    Gradient of a function.

    :param SympyFn.SympyFn sympyFn:
        :class:`SympyFn` object.

    :return:
        Vector function that is the result of partial differentiations with
        respect to the parameters of the :class:`SympyFn` object. The gradient
        components are in the same order as the parameters of the
        :class:`SympyFn` object.

    :rtype: :class:`VectorFn`
    """
    if (not(isinstance(sympyFn, SympyFn))):
        raise TypeError()

    sympyParams = sympyFn.parameters()
    sympyExpr = sympyFn.expression()
    moduleForLambdify = sympyFn.module()

    diffs = tuple(sympyExpr.diff(param, modules = moduleForLambdify) for param in sympyParams)

    return VectorFn([SympyFn(sympyParams, diff, moduleForLambdify) for diff in diffs])

def getColumnVectors(M):
    r"""
    Gets the column vectors of a matrix.

    :param M:
        Matrix of column vectors.

    :return:
        List of vectors that are column vectors of ``M``.

    :rtype: list
    """
    if (not(isinstance(M, numpy.matrix))):
        raise TypeError()

    return [numpy.array(vec).flatten() for vec in M.T]

def getRowVectors(M):
    r"""
    Gets the row vectors of a matrix.

    :param M:
        Matrix of row vectors.

    :return:
        List of vectors that are row vectors of ``M``.

    :rtype: list
    """
    if (not(isinstance(M, numpy.matrix))):
        raise TypeError()

    return [numpy.array(vec).flatten() for vec in M]

def polarCoords(v):
    r"""
    Converts a 2D vector in Cartesian coordinates to polar coordinates.

    :param numpy.ndarray v:
        2D vector in Cartesian coordinates.

    :return:
        Polar coordinates, where radius is positive, and angle is in the
        interval :math:`[0, 2 \pi)`.

    :rtype: pair of float

    :exception ValueError:
        Vector is not two-dimensional.

    :exception ValueError:
        Magnitude is zero.
    """
    if (not(isVector(v))):
        raise TypeError()

    if (v.shape[0] != 2):
        raise ValueError('Vector is not two-dimensional.')

    radius = magnitude(v)

    if (radius == 0):
        raise ValueError('Magnitude is zero.')

    angle = numpy.arccos(v[0] / radius)

    if (v[1] < 0):
        angle = -angle + 2 * numpy.pi

    return (radius, angle)

def polarToVector(radius, angle):
    r"""
    Converts polar coordinates to a 2D vector in Cartesian coordinates.

    :param float radius:
        Radius.

    :param float angle:
        Angle.

    :return:
        2D vector in Cartesian coordinates.

    :rtype: ``numpy.ndarray``
    """
    radius = float(radius)
    angle = float(angle)

    return numpy.array([radius * numpy.cos(angle), radius * numpy.sin(angle)])

def jacobian(vectorFn, asColumnVectors):
    r"""
    Jacobian of a vector function.

    :param VectorFn vectorFn:
        Vector function.

    :param bool asColumnVectors:
        ``True`` if the gradients of the components of ``vectorFn`` are to be
        column vectors in the Jacobian matrix; ``False`` if they are to be row
        vectors.

    :return:
        Jacobian matrix.

    :rtype: ``numpy.matrix`` of :class:`SympyFn.SympyFn` objects.
    """
    if (not(isinstance(vectorFn, VectorFn))):
        raise TypeError()

    asColumnVectors = bool(asColumnVectors)

    # Take the gradient of each component.

    gradients = []

    for cmpt in vectorFn:
        gradients.append(numpy.matrix(gradient(cmpt).components()))

    # Form the Jacobian matrix as row vectors.
    jacobianAsRowVecs = numpy.vstack(gradients)

    if (asColumnVectors):
        return jacobianAsRowVecs.T
    else:
        return jacobianAsRowVecs

def sympyOfFnMatrix(numpyMatrix):
    r"""
    Converts a non-empty Numpy matrix of :class:`SympyFn.SympyFn` objects to a
    Sympy matrix of Sympy expressions.

    :param numpy.matrix numpyMatrix:
        Numpy matrix of :class:`SympyFn.SympyFn` objects.

    :return:
        Pair, where first component is tuple of parameters as Sympy symbols,
        and second component is Sympy matrix of Sympy expressions.

    :rtype: tuple
    """
    if (not(isinstance(numpyMatrix, numpy.matrix))):
        raise TypeError()

    for elt in numpy.nditer(numpyMatrix, ['refs_ok']):
        if (not(isinstance(elt.tolist(), SympyFn))):
            raise TypeError()

    rows = []

    for row in getRowVectors(numpyMatrix):
        rows.append(sympy.Matrix([[sympyFn.expression() for sympyFn in row]]))

    return (numpyMatrix[0, 0].parameters(), sympy.Matrix(rows))

def fnMatrixOfSympy(sympyParams, sympyMatrix, module):
    r"""
    Converts a non-empty Sympy matrix of Sympy expressions to a Numpy matrix of
    :class:`SympyFn.SympyFn` objects.

    :param iterable sympyParams:
        Sequence of parameters as Sympy symbols.

    :param sympyMatrix:
        Sympy matrix of Sympy expressions.

    :param module:
        Module to use for Sympy's lambdify.

    :return:
        Numpy matrix of :class:`SympyFn.SympyFn` objects.

    :rtype: ``numpy.matrix``
    """
    sympyParams = tuple(sympyParams)
    numpyMatrix = numpy.matlib.empty((sympyMatrix.shape[0], sympyMatrix.shape[1]), dtype = SympyFn)

    for rowIdx in range(numpyMatrix.shape[0]):
        for colIdx in range(numpyMatrix.shape[1]):
            numpyMatrix[rowIdx, colIdx] = SympyFn(sympyParams, sympyMatrix[rowIdx, colIdx], module)

    return numpyMatrix
