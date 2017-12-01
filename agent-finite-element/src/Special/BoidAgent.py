r"""
Agent-based modeling of boids.
"""

import abc
import functools
import numpy
import operator
import scipy.optimize
import weakref

import Vector

from warnings import warn

from EventBased import Event
from Float import nearlyEqual
from List import AssocList
from Vector import isVector

ABC = abc.ABCMeta('ABC', (object,), {'__slots__': ()})

class TimepointProgressed(Event):
    r"""
    Event for notifying the boids that the timepoint has progressed.
    """

    def notify(self, oldTimepoint, newTimepoint):
        r"""
        Overrides :func:`EventBased.Event.notify`.

        See :func:`attach` for the notification procedure.

        :param float oldTimepoint:
            Old timepoint.

        :param float newTimepoint:
            New timepoint.

        :return:
            See :func:`EventBased.Event.notify`.
        """
        for boid, (checkSurroundings, (adjustTrajectory,)) in self._handlers:
            checkSurroundings(boid, oldTimepoint, newTimepoint)

        for boid, (checkSurroundings, (adjustTrajectory,)) in self._handlers:
            adjustTrajectory(boid, oldTimepoint, newTimepoint)

    def attach(self, boid, checkSurroundingsHandler, adjustTrajectoryHandler):
        r"""
        Overrides :func:`EventBased.Event.attach`.

        :param Boid boid:
            New boid that is to be registered.

        :param callable checkSurroundingsHandler:
            Binary function that is a member of ``boid``, where first parameter
            is old timepoint, and second parameter is new timepoint.

            The notified boid is to check its surroundings to decide on any
            necessary adjustments in its trajectory when
            ``adjustTrajectoryHandler`` is called.

        :param callable adjustTrajectoryHandler:
            Binary function that is a member of ``boid``, where first parameter
            is old timepoint, and second parameter is new timepoint.  It is
            called after all boids' ``checkSurroundingsHandler`` have been
            called.

            The notified boid is to adjust its trajectory based on the
            surroundings determined through ``checkSurroundingsHandler``.

        :return:
            See :func:`EventBased.Event.attach`.
        """
        if (not(isinstance(boid, Boid))):
            raise TypeError()

        Event.attach(self, boid, checkSurroundingsHandler, adjustTrajectoryHandler)

class BoidFlock(object):
    r"""
    Flock of boids.

    **Data members:**
        ``_boids``: List of boids.

        ``_timepoint``: Timepoint of the flock.

        ``_timepointProgressedEvent``: :class:`TimepointProgressed`.
    """

    def __init__(self, initialTime = 0.0):
        r"""
        :param float initialTime:
            Arbitrary initial time.
        """
        self._boids = []
        self._timepoint = float(initialTime)
        self._timepointProgressedEvent = TimepointProgressed()

    def _addBoid(self, boid):
        r"""
        Adds a boid.

        :param Boid boid:
            Boid to add.

        :return:
            ``None``

        :exception ValueError:
            Boid is already in the flock.
        """
        if (not(isinstance(boid, Boid))):
            raise TypeError()

        if (len([existingBoid for existingBoid in self._boids if existingBoid is boid]) > 0):
            raise ValueError('Boid is already in the flock.')

        self._boids.append(boid)
        self.timepointProgressedEvent().attach(boid, type(boid)._checkSurroundings, type(boid)._adjustTrajectory)

    def boids(self):
        r"""
        Boids in the flock.

        :return:
            Boids in the flock.

        :rtype: sequence of :class:`Boid` objects
        """
        return (boid for boid in self._boids)

    def timepointProgressedEvent(self):
        r"""
        :return: :class:`TimepointProgressed` event.
        """
        return self._timepointProgressedEvent

    def timepoint(self):
        r"""
        Current timepoint of the flock.

        :return:
            Timepoint.

        :rtype: float
        """
        return self._timepoint

    def progressTimepoint(self, progressionLength):
        r"""
        Progresses the timepoint of the flock.

        It first sends the :class:`TimepointProgressed` event to listening
        boids. After event notifications, the flock then updates its timepoint.

        :param float progressionLength:
            Length, as positive ``float``, of the timepoint progression.

        :return:
            ``None``

        :exception ValueError:
            Timepoint progression is not positive.
        """
        progressionLength = float(progressionLength)

        if (not(progressionLength > 0)):
            raise ValueError('Timepoint progression is not positive.')

        oldTimepoint = self.timepoint()
        newTimepoint = oldTimepoint + progressionLength

        self.timepointProgressedEvent().notify(oldTimepoint, newTimepoint)
        self._timepoint = newTimepoint

    def averagePosition(self):
        r"""
        Average position of the flock. It is defined as

        .. math::
            \mathbf{z}_{\text{avg}} = \frac{1}{N} \sum_{i = 1}^{N} \mathbf{z}_{i}

        where :math:`\mathbf{z}_{i}` is position of a boid, and :math:`N` is
        number of boids.

        :return:
            Average position.

        :rtype: numpy.ndarray
        """
        boids = list(self.boids())
        return 1.0 / len(boids) * functools.reduce(operator.add, [boid.position() for boid in boids])

    def averageVelocity(self):
        r"""
        Average velocity of the flock. It is defined as

        .. math::
            \frac{1}{N} \sum_{i = 1}^{N} \mathbf{v}_{i}

        where :math:`\mathbf{v}_{i}` is velocity of a boid, and :math:`N` is
        number of boids.

        :return:
            Average velocity.

        :rtype: numpy.ndarray
        """
        boids = list(self.boids())
        return 1.0 / len(boids) * functools.reduce(operator.add, [boid.velocity() for boid in boids])

    def averageWidth(self):
        r"""
        Average width (i.e., average radius) of the flock. It is defined as

        .. math::
            \frac{1}{N} \sum_{i = 1}^{N} |\mathbf{z}_{i} - \mathbf{z}_{\text{avg}}|

        where :math:`\mathbf{z}_{\text{avg}}` is average position of the flock
        (see :func:`averagePosition`), :math:`\mathbf{z}_{i}` is position of a
        boid, and :math:`N` is number of boids.

        :return:
            Average width.

        :rtype: numpy.ndarray
        """
        boids = list(self.boids())
        return 1.0 / len(boids) * functools.reduce(operator.add, [Vector.magnitude(boid.position() - self.averagePosition()) for boid in boids])

class BoidFlockCsa2d(BoidFlock):
    r"""
    Flock of boids that follow the CSA model in two-dimensional space.

    It uses the following utility function, :math:`f`, where :math:`C`,
    :math:`S`, and :math:`A` are the CSA parameters that determine the relative
    importance of cohesion, separation, and alignment, respectively.

    .. math::
        f(\mathbf{v}; \mathbf{z}, \mathbf{z}_{i}, \mathbf{V}) =
            C \frac{\mathbf{v} \cdot (\Delta \mathbf{z})_{\text{avg}}}{|\mathbf{v}| |(\Delta \mathbf{z})_{\text{avg}}|}
            + A \frac{\mathbf{v} \cdot \mathbf{V}}{|\mathbf{v}| |\mathbf{V}|}
            - S \frac{\mathbf{v} \cdot (\Delta \mathbf{z})_{\text{min}}}{|\mathbf{v}| |(\Delta \mathbf{z})_{\text{min}}|^{3}}

    :math:`\mathbf{v}` is velocity of the boid of interest and can be varied in
    its direction only. :math:`\Delta \mathbf{z}` is displacement from the boid
    of interest to a local boid. :math:`(\Delta \mathbf{z})_{\text{avg}}` is
    average displacement from the boid of interest to local boids.
    :math:`(\Delta \mathbf{z})_{\text{min}}` is displacement from the boid of
    interest to a local boid such that the distance is the shortest.
    :math:`\mathbf{V}` is average velocity of local boids. Mathematically,

    .. math::
        (\Delta \mathbf{z})_{\text{avg}} &= \frac{1}{N} \sum_{i = 1}^{N} (\mathbf{z}_{i} - \mathbf{z}) \\[2mm]
        (\Delta \mathbf{z})_{\text{min}} &= \arg\min_{\Delta \mathbf{z}} |\Delta \mathbf{z}| \\[2mm]
        \mathbf{V} &= \frac{1}{N} \sum_{i = 1}^{N} \mathbf{v}_{i}

    where :math:`i` iterates the local boids, :math:`N` is number of local
    boids, :math:`\mathbf{z}` is position of the boid of interest,
    :math:`\mathbf{z}_{i}` is position of a local boid, and
    :math:`\mathbf{v}_{i}` is velocity of a local boid. Local boids refer to
    other boids that are nearby and do not include the boid of interest.

    **Data members:**
        ``_csaParams``: 3-tuple of the CSA parameters (:math:`C`, :math:`S`,
        and :math:`A`, in order).
    """

    def __init__(self, paramC, paramS, paramA, initialTime = 0.0):
        r"""
        :param float paramC:
            Cohesion parameter of the CSA model.

        :param float paramS:
            Separation parameter of the CSA model.

        :param float paramA:
            Alignment parameter of the CSA model.

        :param float initialTime:
            Arbitrary initial time.
        """
        BoidFlock.__init__(self, float(initialTime))

        self._csaParams = (float(paramC), float(paramS), float(paramA))

    def csaParams(self):
        r"""
        CSA parameters.

        :return:
            CSA parameters, in order, as a 3-tuple.

        :rtype: tuple
        """
        return self._csaParams

class Boid(ABC):
    r"""
    Virtual bird.

    **Data members:**
        ``_flock``: Weak reference to the :class:`BoidFlock` object that the
        boid is a member of.

        ``_position``: Position vector of the boid.

        ``_velocity``: Velocity vector of the boid.

        ``_isLocalBoidFn``: Binary function for determining local boids.
    """

    @abc.abstractmethod
    def _checkSurroundings(self, oldTimepoint, newTimepoint):
        r"""
        ``[pure virtual]`` Asks the boid to check its surroundings.

        :param float oldTimepoint:
            Old timepoint.

        :param float newTimepoint:
            New timepoint.

        :return:
            ``None``
        """
        pass

    @abc.abstractmethod
    def _adjustTrajectory(self, oldTimepoint, newTimepoint):
        r"""
        ``[pure virtual]`` Asks the boid to adjust its trajectory.

        :param float oldTimepoint:
            Old timepoint.

        :param float newTimepoint:
            New timepoint.

        :return:
            ``None``
        """
        pass

    def __init__(self, flock, position, velocity, isLocalBoidFn):
        r"""
        :param BoidFlock flock:
            Flock of boids that the new boid is a member of.

        :param numpy.ndarray position:
            Initial position vector.

        :param numpy.ndarray velocity:
            Initial velocity vector.

        :param callable isLocalBoidFn:
            Binary function, where first parameter is this boid, and second
            parameter is another boid. It returns ``True`` if the other boid is
            considered to be local to this boid; ``False`` if othewise.

        :exception ValueError:
            Position and velocity vectors do not have the same dimension.
        """
        if (not(isinstance(flock, BoidFlock))):
            raise TypeError()

        if (not(callable(isLocalBoidFn))):
            raise TypeError()

        if (not(isVector(position))):
            raise TypeError()

        if (not(isVector(velocity))):
            raise TypeError()

        if (position.shape[0] != velocity.shape[0]):
            raise ValueError('Position and velocity vectors do not have the same dimension.')

        self._flock = weakref.ref(flock)
        self._position = position.copy()
        self._velocity = velocity.copy()
        self._isLocalBoidFn = isLocalBoidFn

        flock._addBoid(self)

    def flock(self):
        r"""
        :class:`BoidFlock` that the boid is a member of.

        :return:
            :class:`BoidFlock` that the boid is a member of.

        :rtype: :class:`BoidFlock`
        """
        flock = (self._flock)()

        if (flock is None):
            raise RuntimeError('Flock is already destroyed.')

        return flock

    def position(self):
        r"""
        Position of the boid.

        :return:
            Position of the boid.

        :rtype: numpy.ndarray
        """
        return self._position.copy()

    def velocity(self):
        r"""
        Velocity of the boid.

        :return:
            Velocity of the boid.

        :rtype: numpy.ndarray
        """
        return self._velocity.copy()

    def localBoids(self):
        r"""
        Boids that are considered to be local.

        :return:
            Local boids.

        :rtype: sequence of :class:`Boid` objects
        """
        otherBoids = filter(lambda boid: not(boid is self), self.flock().boids())
        return (otherBoid for otherBoid in otherBoids if (self._isLocalBoidFn(self, otherBoid)))

class BoidCsa2d(Boid):
    r"""
    Boid that follows the CSA model in two-dimensional space.

    For the CSA model and the utility function, see :class:`BoidFlockCsa2d`.

    **Data members:**
        ``__cache``: Cache.
    """

    def _utilityFunction(self, velocityAngle):
        r"""
        Calculates the value of the utility function, which is defined in
        :class:`BoidFlockCsa2d` but rewritten using velocity angle.

        :param float velocityAngle:
            Angle, within :math:`[0, 2 \pi)`, of the velocity counterclockwise
            from the positive :math:`x`-axis.

        :return:
            Value of the utility function.

        :rtype: float

        :exception ValueError:
            Angle of the velocity is not greater than or equal to 0 and less
            than 2 * pi.
        """
        velocityAngle = float(velocityAngle)

        if (velocityAngle < 0 or nearlyEqual(velocityAngle, 2 * numpy.pi) or velocityAngle > 2 * numpy.pi):
            raise ValueError('Angle of the velocity is not greater than or equal to 0 and less than 2 * pi.')

        paramC, paramS, paramA = self.flock().csaParams()
        localBoids = list(self.localBoids())

        # Displacements from this boid to local boids as an association list.
        localBoidDisplacements = [
            localBoid.position() - self.position()
            for localBoid in localBoids
        ]

        # Average displacement to local boids.
        dz_avg = 1.0 / len(localBoids) * functools.reduce(operator.add, localBoidDisplacements)

        # Displacement to a local boid with the shortest distance.

        argminIndex = numpy.argmin([Vector.magnitude(displacement) for displacement in localBoidDisplacements])
        dz_min = localBoidDisplacements[argminIndex]

        # Average velocity of local boids.
        V = 1.0 / len(localBoids) * functools.reduce(operator.add, [localBoid.velocity() for localBoid in localBoids])

        return (
            paramC * numpy.cos(velocityAngle - Vector.polarCoords(dz_avg)[1]) \
            + paramA * numpy.cos(velocityAngle - Vector.polarCoords(V)[1]) \
            - paramS * numpy.cos(velocityAngle - Vector.polarCoords(dz_min)[1]) / Vector.magnitude(dz_min) ** 2
        )

    def _checkSurroundings(self, oldTimepoint, newTimepoint):
        r"""
        Overrides :func:`Boid._checkSurroundings`.

        Prepares for adjusting its trajectory by determining the new velocity
        angle. It does not adjust its trajectory until
        :func:`_adjustTrajectory` is automatically triggered by the
        :class:`TimepointProgressed` event.

        :exception RuntimeError:
            Maximization of the utility function terminated unsuccessfully.
        """
        if (len(list(self.localBoids())) == 0):
            # Velocity angle remains the same.
            self.__cache = Vector.polarCoords(self.velocity())[1]
        else:
            optimizationResult = scipy.optimize.minimize_scalar(
                lambda velocityAngle: -self._utilityFunction(velocityAngle),
                bounds = (0, 2 * numpy.pi),
                method = 'Bounded'
            )

            if (not(optimizationResult.success)):
                raise RuntimeError('Maximization of the utility function terminated unsuccessfully.')

            newVelocityAngle = optimizationResult.x

            if (nearlyEqual(newVelocityAngle, 2 * numpy.pi)):
                newVelocityAngle = 0

            self.__cache = newVelocityAngle

    def _adjustTrajectory(self, oldTimepoint, newTimepoint):
        r"""
        Overrides :func:`Boid._adjustTrajectory`.
        """
        # When this function is triggered by the TimepointProgressed event, the
        # cache contains the new velocity angle.

        newVelocityAngle = self.__cache
        newVelocity = Vector.polarToVector(Vector.magnitude(self.velocity()), newVelocityAngle)
        newPosition = self.position() + newVelocity * (newTimepoint - oldTimepoint)

        self._velocity = newVelocity
        self._position = newPosition

        # Clear cache.
        self.__cache = None

    def __init__(self, flock, position, velocity, isLocalBoidFn):
        r"""
        :param BoidFlock flock:
            Flock of boids that the new boid is a member of.

        :param numpy.ndarray position:
            Initial position vector.

        :param numpy.ndarray velocity:
            Initial velocity vector.

        :param callable isLocalBoidFn:
            See :func:`Boid.__init__`.
        """
        Boid.__init__(self, flock, position, velocity, isLocalBoidFn)

        self.__cache = None
