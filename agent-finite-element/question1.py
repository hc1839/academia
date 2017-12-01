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
import matplotlib.animation as animation
import matplotlib.pyplot as pyplot
import numpy
import scipy.optimize

import Vector

from BoidAgent import BoidFlock, Boid, BoidFlockCsa2d, BoidCsa2d
from PrintedLine import printWithCr

# For reproducibility during testing.
numpy.random.seed(2)

SEPARATION_PARAMETER = 1.0 / 4.0
ALIGNMENT_PARAMETER = 5.0
LOCALITY_RADIUS = 1.0

# String prepended to the operation name when printing to standard output.
CURRENT_OPERATION_INDICATOR = 'Current operation: '

def isLocalBoid(boid1, boid2):
    r"""
    Determines whether two boids are local to each other by distance.

    :param Boid boid1:
        A boid.

    :param Boid boid2:
        Another boid.

    :return:
        ``True`` if the distance between ``boid1`` and ``boid2`` is less than
        ``LOCALITY_RADIUS``; ``False`` if otherwise.

    :rtype: bool
    """
    if (not(isinstance(boid1, Boid))):
        raise TypeError()

    if (not(isinstance(boid2, Boid))):
        raise TypeError()

    return Vector.magnitude(boid2.position() - boid1.position()) < LOCALITY_RADIUS

def varianceOfAverageWidths(paramC):
    r"""
    Variance of average widths of a flock of four boids evolving from Timepoint
    0 to 5 in increments of 0.1.

    It is used for determining a cohesion parameter of the CSA model by
    training a flock of four boids. Separation and alignment parameters are
    constants dictated by ``SEPARATION_PARAMETER`` and ``ALIGNMENT_PARAMETER``,
    respectively.

    :param float paramC:
        Cohesion parameter.

    :return:
        Variance of average flock widths.

    :rtype: float
    """
    paramC = float(paramC)

    # Parameters.
    timeStepLength = 0.1
    timeStepCount = int(round(5.0 / timeStepLength))
    initialVelocity = Vector.new(1, 1)

    flock = BoidFlockCsa2d(paramC, SEPARATION_PARAMETER, ALIGNMENT_PARAMETER)

    # Construct four boids.
    for initialPosition in itertools.product([0.25, 0.75], [0.25, 0.75]):
        BoidCsa2d(flock, Vector.new(*initialPosition), initialVelocity, isLocalBoid)

    averageWidths = [flock.averageWidth()]

    # Evolve and get average widths of the flock.
    for stepIdx in range(timeStepCount):
        flock.progressTimepoint(timeStepLength)
        averageWidths.append(flock.averageWidth())

    return numpy.var(averageWidths, ddof = 0)

def evolveFlock(flock, timeStepLength, timeStepCount, initialTimepoint = 0.0):
    r"""
    Evolve the flock, and return some of its properties for plotting and
    animation.

    :param BoidAgent.BoidFlock flock:
        Flock to evolve. It is modified.

    :param float timeStepLength:
        Length of each time step.

    :param int timeStepCount:
        Number of time steps.

    :param float initialTimepoint:
        Initial timepoint.

    :return:
        Pair of dictionaries, where first component is boid attributes to be
        animated, and second component is flock attributes to be animated.

        For the boid attributes, the following keys of the dictionary are
        returned.

            * ``timepoints``
            * ``positions``

        For the flock attributes, the following keys of the dictionary are
        returned.

            * ``timepoints``
            * ``positions``
            * ``velocities``
            * ``widths``

    :rtype: tuple of dict
    """
    if (not(isinstance(flock, BoidFlock))):
        raise TypeError()

    timeStepLength = float(timeStepLength)
    timeStepCount = int(timeStepCount)
    initialTimepoint = float(initialTimepoint)

    boidAttrs = {
        'timepoints': [initialTimepoint],
        'positions': [numpy.array([boid.position() for boid in flock.boids()])]
    }

    flockAttrs = {
        'timepoints': [initialTimepoint],
        'positions': [flock.averagePosition()],
        'velocities': [flock.averageVelocity()],
        'widths': [flock.averageWidth()]
    }

    for stepIdx in range(timeStepCount):
        flock.progressTimepoint(timeStepLength)
        initialTimepoint += timeStepLength

        boidAttrs['timepoints'].append(initialTimepoint)
        boidAttrs['positions'].append(numpy.array([boid.position() for boid in flock.boids()]))

        flockAttrs['timepoints'].append(initialTimepoint)
        flockAttrs['positions'].append(flock.averagePosition())
        flockAttrs['velocities'].append(flock.averageVelocity())
        flockAttrs['widths'].append(flock.averageWidth())

    return (boidAttrs, flockAttrs)

def animateFlockAttrs(boidAttrs, flockAttrs):
    r"""
    Animates boid and flock attributes.

    :param dict boidAttrs:
        Boid attributes to animate.

        The following keys are required, with values as lists:

            * ``timepoints``
            * ``positions``

    :param dict flockAttrs:
        Flock attributes to animate.

        The following keys are required, with values as lists:

            * ``timepoints``
            * ``positions``
            * ``velocities``
            * ``widths``
    :return:
        ``None``
    """
    boidAttrs = dict(boidAttrs)
    flockAttrs = dict(flockAttrs)

    timeStepLength = flockAttrs['timepoints'][1] - flockAttrs['timepoints'][0]
    timeStepCount = len(flockAttrs['timepoints'])

    def dataGen():
        for idx in range(len(flockAttrs['timepoints'])):
            yield {
                'timepoint': flockAttrs['timepoints'][idx],
                'boid-positions': boidAttrs['positions'][idx],
                'flock-position': flockAttrs['positions'][idx],
                'flock-velocity': flockAttrs['velocities'][idx],
                'flock-width': flockAttrs['widths'][idx]
            }

    def init():
        ax['boid-positions'].set_xlim(-10.0, 20.0)
        ax['boid-positions'].set_ylim(-10.0, 20.0)

        ax['flock-width'].set_xlim(0.0, timeStepLength * timeStepCount)
        ax['flock-width'].set_ylim(0.0, 10.0)

        ax['flock-trajectory'].set_xlim(-10.0, 20.0)
        ax['flock-trajectory'].set_ylim(-10.0, 20.0)

        del timepoints[:]
        del flockWidthData[:]

    def animate(data):
        timepoints.append(data['timepoint'])

        boidPositionTimepointLabel.set_text('Time = %.2f'%(data['timepoint']))
        curve['boid-positions'].set_data(
            data['boid-positions'][:, 0],
            data['boid-positions'][:, 1]
        )

        flockWidthData.append(data['flock-width'])
        curve['flock-width'].set_data(timepoints, flockWidthData)

        flockTrajectoryTimepointLabel.set_text('Time = %.2f'%(data['timepoint']))
        curve['flock-position'].set_data(
            data['flock-position'][0],
            data['flock-position'][1]
        )
        curve['flock-velocity'].set_data(
            data['flock-velocity'][0],
            data['flock-velocity'][1]
        )

    fig = pyplot.figure()

    ax = {
        'boid-positions': fig.add_subplot(1, 3, 1),
        'flock-width': fig.add_subplot(1, 3, 2),
        'flock-trajectory': fig.add_subplot(1, 3, 3)
    }

    ax['boid-positions'].set_xlabel(r'$x$')
    ax['boid-positions'].set_ylabel(r'$y$')
    ax['boid-positions'].set_title(r'Boid Position')

    ax['flock-width'].set_xlabel(r'Time')
    ax['flock-width'].set_ylabel(r'Average Flock Width')
    ax['flock-width'].set_title(r'Flock Width')

    ax['flock-trajectory'].set_xlabel(r'$x$')
    ax['flock-trajectory'].set_ylabel(r'$y$')
    ax['flock-trajectory'].set_title(r'Flock Trajectory')

    curve = {
        'boid-positions': ax['boid-positions'].plot([], [], 'bo', ms = 2)[0],
        'flock-width': ax['flock-width'].plot([], [], 'b-', lw = 1)[0],
        'flock-position': ax['flock-trajectory'].plot([], [], 'bo', ms = 3, label = 'Position')[0],
        'flock-velocity': ax['flock-trajectory'].plot([], [], 'ro', ms = 3, label = 'Velocity')[0]
    }

    for elt in ax.values():
        elt.grid()
        elt.legend()

    boidPositionTimepointLabel = ax['boid-positions'].text(0.05, 0.05, '', transform = ax['boid-positions'].transAxes)
    flockTrajectoryTimepointLabel = ax['flock-trajectory'].text(0.05, 0.05, '', transform = ax['flock-trajectory'].transAxes)

    # Used in the animate function.
    timepoints = []

    # Flock widths.
    flockWidthData = []

    funcAnimation = animation.FuncAnimation(
        fig,
        animate,
        dataGen,
        blit = False,
        interval = 100,
        repeat = True,
        init_func = init
    )

    pyplot.tight_layout()

    try:
        pyplot.show()
    except AttributeError:
        pass

if (__name__ == '__main__'):

    # Train a flock of four boids.

    printWithCr(CURRENT_OPERATION_INDICATOR + 'Training a test flock...')

    optimizationResult = scipy.optimize.minimize_scalar(
        varianceOfAverageWidths,
        bounds = (0.1, 10),
        method = 'Bounded'
    )

    if (not(optimizationResult.success)):
        raise RuntimeError('Minimization of the variance of average widths terminated unsuccessfully.')

    # Cohesion parameter.
    COHESION_PARAMETER = optimizationResult.x

    # Construct a flock of 50 boids.

    printWithCr(CURRENT_OPERATION_INDICATOR + 'Constructing a flock of 50 boids...')

    flock = BoidFlockCsa2d(COHESION_PARAMETER, SEPARATION_PARAMETER, ALIGNMENT_PARAMETER)

    for idx in range(50):
        initialPosition = Vector.new(5.0 * numpy.random.random(), 5.0 * numpy.random.random())
        initialVelocity = Vector.new(1, 1) + 1e-2 * Vector.new(numpy.random.random(), numpy.random.random())

        BoidCsa2d(flock, initialPosition, initialVelocity, isLocalBoid)

    timeStepLength = 0.05
    timeStepCount = 200
    initialTimepoint = 0.0

    printWithCr(CURRENT_OPERATION_INDICATOR + 'Evolving the flock...')
    boidAttrs, flockAttrs = evolveFlock(flock, timeStepLength, timeStepCount, initialTimepoint)

    print()
    print('Complete.')

    animateFlockAttrs(boidAttrs, flockAttrs)
