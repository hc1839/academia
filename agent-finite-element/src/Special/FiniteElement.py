r"""
Finite element construction and representation.

All indices are zero-based.
"""

import abc
import numpy
import sympy
import weakref

import Vector

from Float import nearlyEqual
from List import AssocList
from SympyFn import SympyFn
from Vector import isVector, VectorFn

ABC = abc.ABCMeta('ABC', (object,), {'__slots__': ()})

class Element(ABC):
    r"""
    Finite element.

    Subclasses must have constructors with the same required arguments as
    :func:`__init__`.

    **Data members:**
        ``_mesh``: See :func:`mesh`.

        ``_nodes``: List of nodes. See :func:`nodes`.

        ``_shapeFns``: List of local shape functions. See :func:`__init__`.

        ``_coordTransformation``: See :func:`coordTransformation`.
    """

    def __init__(self, mesh, nodes, shapeFns):
        r"""
        :param Mesh mesh:
            Mesh that the element belongs to.

        :param iterable nodes:
            Sequence of :class:`Node` objects. Order is not important. Local
            indices are determined by element type and node positions.

        :param iterable shapeFns:
            Sequence of shape functions, as :class:`SympyFn.SympyFn` objects,
            ordered by local index. The shape functions are in parent element
            coordinates.

        :exception ValueError:
            Number of nodes is zero.

        :exception ValueError:
            Number of shape functions is zero.

        :exception ValueError:
            Number of shape functions is not the same as number of nodes.

        :exception ValueError:
            Nodes do not have the same dimensionality.

        :exception ValueError:
            Shape functions do not have the same parameters.

        :exception ValueError:
            Shape functions do not have the same dimensionality as the nodes.
        """
        if (not(isinstance(mesh, Mesh))):
            raise TypeError()

        nodes = list(nodes)

        if (len(list(filter(lambda elt: not(isinstance(elt, Node)), nodes))) > 0):
            raise TypeError()

        shapeFns = list(shapeFns)

        if (len(list(filter(lambda elt: not(isinstance(elt, SympyFn)), shapeFns))) > 0):
            raise TypeError()

        if (len(nodes) == 0):
            raise ValueError('Number of nodes is zero.')

        if (len(shapeFns) == 0):
            raise ValueError('Number of shape functions is zero.')

        if (len(nodes) != len(shapeFns)):
            raise ValueError('Number of shape functions is not the same as number of nodes.')

        if (len(frozenset(node.position().shape[0] for node in nodes)) != 1):
            raise ValueError('Nodes do not have the same dimensionality.')

        if (len(frozenset(shapeFn.parameters() for shapeFn in shapeFns)) != 1):
            raise ValueError('Shape functions do not have the same parameters.')

        if (not(
            len(frozenset(len(list(shapeFn.parameters())) for shapeFn in shapeFns)) == 1 and
            len(list(shapeFns[0].parameters())) == nodes[0].position().shape[0]
        )):
            raise ValueError('Shape functions do not have the same dimensionality as the nodes.')

        self._mesh = weakref.ref(mesh)
        self._nodes = nodes
        self._shapeFns = shapeFns
        self._coordTransformation = None

        mesh._addElement(self)

    def mesh(self):
        r"""
        :return:
            Mesh that the element belongs to.

        :rtype: :class:`Mesh`
        """
        objref = (self._mesh)()

        if (objref is None):
            raise RuntimeError('Mesh is already destroyed.')

        return objref

    def nodes(self):
        r"""
        Nodes of the element.

        :return:
            Nodes of the element.

        :rtype: sequence of :class:`Node` objects
        """
        return self._nodes.copy()

    @abc.abstractmethod
    def getNode(self, localIndex):
        r"""
        ``[pure virtual]`` Gets a node by local index with respect to the
        element.

        :param int localIndex:
            Local index.

        :return:
            Node at ``localIndex``.

        :rtype: :class:`Node`

        :exception ValueError:
            Invalid local index.
        """
        pass

    def getShapeFn(self, localIndex):
        r"""
        Gets a shape function, in parent element coordinates, by local index
        with respect to the element.

        :param int localIndex:
            Local index.

        :return:
            Shape function at ``localIndex``.

        :rtype: :class:`SympyFn.SympyFn`

        :exception ValueError:
            Invalid local index.
        """
        localIndex = int(localIndex)

        try:
            return self._shapeFns[localIndex]
        except IndexError:
            raise ValueError('Invalid local index.')

    def coordTransformation(self, module):
        r"""
        Coordination transformation from parent element to mesh element.

        It is newly constructed only during the first invocation of this
        function.

        :param module:
            Module to use for Sympy's lambdify.

        :return:
            :class:`Vector.VectorFn` object that performs the transformation.

        :rtype: :class:`Vector.VectorFn`
        """
        if (self._coordTransformation is None):
            orderedNodePositions = []    # Many row vectors.
            orderedShapeFnExprs = []    # One row vector.

            # Get node positions and shape function expressions as row vectors.
            for localIndex in range(len(list(self.nodes()))):
                orderedNodePositions.append(self.getNode(localIndex).position())
                orderedShapeFnExprs.append(self.getShapeFn(localIndex).expression())

            # Matrix multiplication of node positions and shape functions as
            # Sympy expressions.
            sympyExprs = numpy.array(
                sympy.Matrix(orderedNodePositions).T.multiply(sympy.Matrix([orderedShapeFnExprs]).T)
            ).flatten()

            parentElementParams = self.getShapeFn(0).parameters()

            self._coordTransformation = VectorFn([
                SympyFn(parentElementParams, sympyExpr, module)
                for sympyExpr in sympyExprs
            ])

        return self._coordTransformation

class Element2d(Element):
    r"""
    2D finite element.
    """

    def __init__(self, mesh2d, nodes, shapeFns):
        r"""
        :param Mesh2d mesh2d:
            2D mesh that the element belongs to.

        :param iterable nodes:
            See :func:`Element.__init__`.

        :param iterable shapeFns:
            See :func:`Element.__init__`.

        :exception ValueError:
            Node positions do not represent a two-dimensional element.
        """
        Element.__init__(self, mesh2d, nodes, shapeFns)

        dimensionalities = frozenset(node.position().shape[0] for node in nodes)

        if (not(len(dimensionalities) == 1 and (2 in dimensionalities))):
            raise ValueError('Node positions do not represent a two-dimensional element.')

class Node(ABC):
    r"""
    Node of a region.

    **Data members:**
        ``_position``: See :func:`position`.

        ``_globalIndex``: See :func:`globalIndex`.

        ``_mesh``: Weak reference to parent mesh. It is initialized to ``None``
        at instance construction. Friend of :func:`Mesh._addElement`.
    """

    def __init__(self, position):
        r"""
        ``[private]`` Friend of :class:`NonDirichletNode` and
        :class:`DirichletNode`.

        :param numpy.ndarray position:
            Position of the node with respect to the global region.
        """
        if (not(isVector(position))):
            raise TypeError()

        position = position.copy()

        self._position = position
        self._globalIndex = None
        self._mesh = None

    def position(self):
        r"""
        :return:
            Position of the node with respect to the global region.

        :rtype: numpy.ndarray
        """
        return self._position.copy()

    def globalIndex(self):
        r"""
        Global index of the node.

        :class:`Element` object should not call this function when the element
        is on a mesh scaffold, since the node may be a placeholder. See
        :func:`NodePlaceholder.globalIndex`.

        :return:
            Global index.

        :rtype: int

        :exception RuntimeError:
            Global index unavailable.
        """
        if (self._globalIndex is None):
            raise RuntimeError('Global index unavailable.')

        return self._globalIndex

    def mesh(self):
        r"""
        Mesh that the node belongs to.

        :return:
            Parent mesh.

        :rtype: :class:`Mesh`

        :exception RuntimeError:
            Node does not belong to a mesh.
        """
        if (self._mesh is None):
            raise RuntimeError('Node does not belong to a mesh.')

        objref = self._mesh()

        if (objref is None):
            raise RuntimeError('Mesh is already destroyed.')

        return objref

class DirichletNode(Node):
    r"""
    Node with Dirichlet boundary condition.
    """

    def __init__(self, position):
        r"""
        :param numpy.ndarray position:
            See :func:`Node.__init__`.
        """
        Node.__init__(self, position)

class NonDirichletNode(Node):
    r"""
    Non-Dirichlet node.
    """

    def __init__(self, position):
        r"""
        :param numpy.ndarray position:
            See :func:`Node.__init__`.
        """
        Node.__init__(self, position)

class NeumannNode(NonDirichletNode):
    r"""
    Node with flux boundary condition.
    """

    def __init__(self, position):
        r"""
        :param numpy.ndarray position:
            See :func:`UndeterminedNode.__init__`.
        """
        Node.__init__(self, position)

class NodePlaceholder(Node):
    r"""
    Placeholder for a node.

    **Data members:**
        ``_nodeType``: See :func:`nodeType`.
    """

    def __init__(self, nodeType, position):
        r"""
        :param type nodeType:
            Type of node that the placeholder is representing. Its constructor
            accepts one required argument, the global node position.

        :param numpy.ndarray position:
            See :func:`Node.__init__`.
        """
        Node.__init__(self, position)

        if (not(isinstance(nodeType, type))):
            raise TypeError()

        self._nodeType = nodeType

    def globalIndex(self):
        r"""
        Overrides :func:`Node.globalIndex`.

        Always raises ``NotImplementedError`` exception.
        """
        raise NotImplementedError()

    def nodeType(self):
        r"""
        :return:
            Type of node that the placeholder is representing.

        :rtype: type
        """
        return self._nodeType

class Mesh(ABC):
    r"""
    Mesh represented by :class:`Element` objects.

    Constructors of subclasses must have no required arguments and be
    accessible to :func:`MeshScaffold.build`.

    **Data members:**
        ``_elements``: List of :class:`Element` objects.

        ``_nonDirichletNodes``: List of non-Dirichlet nodes ordered in
        increasing global node indices. It is initialized to ``None`` on
        instance construction.
    """

    def _addElement(self, element):
        r"""
        Adds an :class:`Element` object.

        Subclasses should override this function to provide additional sanity
        checks. Overridden function must be a friend of
        :func:`MeshScaffold.build` or its corresponding subclass.

        :param Element element:
            Element. Nodes of the element must not be placeholders.

        :return:
            ``None``

        :exception ValueError:
            Element exists.

        :exception ValueError:
            A node of the element is a placeholder.
        """
        if (not(isinstance(element, Element))):
            raise TypeError()

        if (len(list(filter(lambda elt: elt is element, self.elements()))) > 0):
            raise ValueError('Element exists.')

        for node in element.nodes():
            if (isinstance(node, NodePlaceholder)):
                raise ValueError('A node of the element is a placeholder.')

        self._elements.append(element)
        self._nonDirichletNodes = None

        for node in element.nodes():
            node._mesh = weakref.ref(self)

    def __init__(self):
        r"""
        ``[protected]`` Friend of :func:`MeshScaffold.build`.
        """
        self._elements = []

    def elements(self):
        r"""
        Elements of the mesh.

        :return:
            Elements.

        :rtype: sequence of :class:`Element` objects
        """
        return (element for element in self._elements)

    def nodes(self):
        r"""
        Nodes of the mesh.

        :return:
            Nodes.

        :rtype: sequence of :class:`Node` objects
        """
        encounteredNodes = []

        # Ensure that a node is in the list once.
        for element in self.elements():
            for node in element.nodes():
                if (len(list(filter(lambda elt: elt is node, encounteredNodes))) == 0):
                    encounteredNodes.append(node)
                    yield node

    def getNonDirichletNodeIndex(self, nonDirichletNode):
        r"""
        Gets the zero-based index of a non-Dirichlet node.

        It is suitable for use as matrix element indices.

        :param NonDirichletNode nonDirichletNode:
            Non-Dirichlet node.

        :return:
            Zero-based index, where indexing is successive and is based on the
            order of increasing global node indices not including Dirichlet
            nodes.

        :rtype: int

        :exception ValueError:
            Node is not a member of this mesh.

        :exception RuntimeError:
            Node cannot be found. It may have been removed without updating.
        """
        if (not(isinstance(nonDirichletNode, NonDirichletNode))):
            raise TypeError()

        if (not(nonDirichletNode.mesh() is self)):
            raise ValueError('Node is not a member of this mesh.')

        if (self._nonDirichletNodes is None):
            self._nonDirichletNodes = sorted(
                [node for node in self.nodes() if (isinstance(node, NonDirichletNode))],
                key = lambda node: node.globalIndex()
            )

        for idx in range(len(self._nonDirichletNodes)):
            if (self._nonDirichletNodes[idx] is nonDirichletNode):
                return idx

        raise RuntimeError('Node cannot be found. It may have been removed without updating.')

class MeshScaffold(Mesh):
    r"""
    Scaffold for constructing :class:`Mesh` objects.

    **Data members:**
        ``_elements``: List of :class:`Element` objects.

        ``_meshType``: See :func:`meshType`.
    """

    def _addElement(self, element):
        r"""
        Adds an :class:`Element` object to the scaffold.

        Constructors for the types of elements must accept three required
        arguments as specified in :func:`Element.__init__`.

        Friend of :func:`Element.__init__`.

        :param Element element:
            Element that contains only :class:`NodePlaceholder` objects for
            representing its nodes.

        :return:
            ``None``

        :exception ValueError:
            Element exists.

        :exception ValueError:
            Nodes of the element are not placeholders.
        """
        if (not(isinstance(element, Element))):
            raise TypeError()

        if (len(list(filter(lambda elt: elt is element, self.elements()))) > 0):
            raise ValueError('Element exists.')

        for node in element.nodes():
            if (not(isinstance(node, NodePlaceholder))):
                raise ValueError('Nodes of the element are not placeholders.')

        self._elements.append(element)

    def __init__(self, meshType):
        r"""
        Instance constructor.

        :param type meshType:
            Type of mesh that the scaffold is building.
        """
        Mesh.__init__(self)

        if (not(isinstance(meshType, type))):
            raise TypeError()

        self._elements = []
        self._meshType = meshType

    def meshType(self):
        r"""
        Type of mesh that the scaffold is building.

        :return:
            Final mesh type.

        :rtype: type
        """
        return self._meshType

    def elements(self):
        r"""
        Overrides :func:`Mesh.elements`.
        """
        return self._elements.copy()

    def build(self, globalNodeIndexer):
        r"""
        Constructs a :class:`Mesh` object from the scaffold.

        Sanity checks should be performed.

        :param callable globalNodeIndexer:
            Ternary function, where first parameter is new :class:`Mesh` object
            that is being constructed, second parameter is sequence of nodes
            currently in the new mesh, and third parameter is newly constructed
            :class:`Node` object that is not a placeholder and lacks a global
            index. It returns an ``int`` that specifies the global index of the
            node.

        :return:
            Instance of a subclass of :class:`Mesh`.

        :rtype: subclass of :class:`Mesh`

        :exception ValueError:
            Newly constructed node is a placeholder.

        :exception ValueError:
            Two node placeholders of different types exist at the same
            position.

        :exception ValueError:
            Nodes do not have unique global indices.
        """
        if (not(callable(globalNodeIndexer))):
            raise TypeError()

        newMesh = (self.meshType())()

        # All nodes of the new mesh. Mapping from unique position vector to
        # node.
        concreteNodesNewMesh = AssocList(lambda a, b: Vector.nearlyEqualVectors(a, b))

        # Construct new elements using concrete nodes without redundancy in
        # global positions.
        for oldElement in self.elements():
            # Nodes of the new element.
            concreteNodesNewElement = []

            for nodePlaceholder in oldElement.nodes():
                nodePlaceholderPosition = nodePlaceholder.position()
                nodePlaceholderType = nodePlaceholder.nodeType()

                # Construct a new or use a reference to an existing node for
                # the new element.
                if (concreteNodesNewMesh.isMember(nodePlaceholderPosition)):
                    concreteNode = concreteNodesNewMesh[nodePlaceholderPosition]

                    # Check that no two nodes of different types exist at the
                    # same position.
                    if (type(concreteNode) != nodePlaceholderType):
                        raise ValueError('Two node placeholders of different types exist at the same position.')
                    else:
                        concreteNodesNewElement.append(concreteNode)
                else:
                    concreteNode = nodePlaceholder.nodeType()(nodePlaceholder.position())

                    if (isinstance(concreteNode, NodePlaceholder)):
                        raise ValueError('Newly constructed node is a placeholder.')

                    # Assign a global index to the node.
                    concreteNode._globalIndex = int(
                        globalNodeIndexer(
                            newMesh, list(zip(*concreteNodesNewMesh))[1] if (len(concreteNodesNewMesh) > 0) else [], concreteNode
                        )
                    )

                    concreteNodesNewElement.append(concreteNode)
                    concreteNodesNewMesh.add(concreteNode.position(), concreteNode)

            shapeFns = [
                oldElement.getShapeFn(localIndex)
                for localIndex in range(len(list(oldElement.nodes())))
            ]
            type(oldElement)(newMesh, concreteNodesNewElement, shapeFns)

        globalIndices = [node.globalIndex() for node in newMesh.nodes()]

        if (len(globalIndices) != len(frozenset(globalIndices))):
            raise ValueError('Nodes do not have unique global indices.')

        return newMesh

class Mesh2d(Mesh):
    r"""
    2D mesh.
    """

    def _addElement(self, element2d):
        r"""
        Overrides :func:`Mesh._addElement`. Friend of
        :func:`MeshScaffold.build`.

        :param Element2d element2d:
            2D element.

        :return:
            See :func:`Mesh._addElement`.
        """
        if (not(isinstance(element2d, Element2d))):
            raise TypeError()

        Mesh._addElement(self, element2d)

    def __init__(self):
        r"""
        ``[protected]`` Friend of :func:`MeshScaffold.build`.
        """
        Mesh.__init__(self)

class ShapeFn(SympyFn):
    r"""
    Shape function.

    **Data members:**
        ``_gradient``: See :func:`gradient`.
    """

    def __init__(self, sympyParams, sympyExpr, module):
        r"""
        See :func:`SympyFn.SympyFn.__init__`.
        """
        SympyFn.__init__(self, sympyParams, sympyExpr, module)

        self._gradient = None

    def gradient(self):
        r"""
        Gradient of the function.

        It is newly determined only during the first invocation of this
        function.

        :return:
            Gradient.

        :rtype: :class:`Vector.VectorFn`
        """
        if (self._gradient is None):
            self._gradient = Vector.gradient(self)

        return self._gradient
