r"""
Test module for Question 1.
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

import Vector
import question1

from BoidAgent import BoidFlock, Boid, BoidFlockCsa2d, BoidCsa2d

COHESION_PARAMETER = 0.750
SEPARATION_PARAMETER = 1.0 / 4.0
ALIGNMENT_PARAMETER = 5.0
LOCALITY_RADIUS = 1.0

class test_question1(unittest.TestCase):
    def test_flockOfOne(self):
        r"""
        Construct a flock of one boid. It should fly straight.
        """
        flock = BoidFlockCsa2d(COHESION_PARAMETER, SEPARATION_PARAMETER, ALIGNMENT_PARAMETER)

        BoidCsa2d(flock, Vector.new(-7.5, 2.5), Vector.new(1, 0.5), question1.isLocalBoid)

        timeStepLength = 0.05
        timeStepCount = 200
        initialTimepoint = 0.0

        boidAttrs, flockAttrs = question1.evolveFlock(flock, timeStepLength, timeStepCount, initialTimepoint)

        print('The solo boid should fly straight.')
        question1.animateFlockAttrs(boidAttrs, flockAttrs)

    def test_flockOfCollidingTwo(self):
        r"""
        Construct a symmetric flock of two boids and make them collide. They
        should bounce away from each other due to the cubic term in the CSA
        utility function.
        """
        flock = BoidFlockCsa2d(COHESION_PARAMETER, SEPARATION_PARAMETER, ALIGNMENT_PARAMETER)

        BoidCsa2d(flock, Vector.new(-7.5, 7.5), Vector.new(1.0, -0.5), question1.isLocalBoid)
        BoidCsa2d(flock, Vector.new(-7.5, 2.5), Vector.new(1.0, 0.5), question1.isLocalBoid)

        timeStepLength = 0.05
        timeStepCount = 300
        initialTimepoint = 0.0

        boidAttrs, flockAttrs = question1.evolveFlock(flock, timeStepLength, timeStepCount, initialTimepoint)

        print('The two boids should fly toward each other and then bounce apart.')
        question1.animateFlockAttrs(boidAttrs, flockAttrs)

    def test_flockOfDistantTwo(self):
        r"""
        Construct a flock two boids distant from each other
        """
        flock = BoidFlockCsa2d(COHESION_PARAMETER, SEPARATION_PARAMETER, ALIGNMENT_PARAMETER)

        BoidCsa2d(flock, Vector.new(-7.5, 10.0), Vector.new(5.0, 0.0), question1.isLocalBoid)
        BoidCsa2d(flock, Vector.new(-7.5, 8.0), Vector.new(1.0, -1.0), question1.isLocalBoid)

        timeStepLength = 0.05
        timeStepCount = 100
        initialTimepoint = 0.0

        boidAttrs, flockAttrs = question1.evolveFlock(flock, timeStepLength, timeStepCount, initialTimepoint)

        print('The two boids should fly independently from each other, with one of them flying faster.')
        question1.animateFlockAttrs(boidAttrs, flockAttrs)

if (__name__ == '__main__'):
    unittest.main()
