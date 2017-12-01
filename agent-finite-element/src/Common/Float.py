r"""
Custom-made floating-point operations.
"""

import sys

def nearlyEqual(a, b, epsilon = sys.float_info.min):
    r"""
    Determines whether two floating-point numbers are considered to be nearly
    equal.

    Adapted from `"The Floating-Point Guide"
    <http://floating-point-gui.de/errors/comparison/>`_.

    :param float a:
        Floating-point number.

    :param float b:
        Floating-point number.

    :param float epsilon:
        Arbitrarily small positive number for testing difference.

    :return:
        ``True`` if the difference between ``a`` and ``b`` is sufficiently small to be considered as nearly equal; ``False`` if otherwise.

    :rtype: bool
    """
    absA = abs(a)
    absB = abs(b)
    diff = abs(a - b)

    if (not(epsilon > 0)):
        raise ValueError('Epsilon is not a positive number.')

    if (a == b):
        return True
    elif (a == 0 or b == 0 or diff < sys.float_info.min):
        return diff < (epsilon * sys.float_info.min)
    else:
        return diff / min(absA + absB, sys.float_info.max) < epsilon
