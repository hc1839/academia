r"""
Facilitates overwriting the printed line.
"""

from __future__ import print_function

import sys

def printWithCr(string, stream = sys.stdout):
    r"""
    Prints a string with a carriage return.

    It remembers the length of the previous string it printed and pads the
    string to be printed with spaces, if necessary, to overwrite the previously
    printed string.

    :param str string:
        String to print.

    :param stream:
        Stream to print to.

    :return:
        ``None``
    """
    string = str(string)

    cntPadding = max([printWithCr.previousStringLength - len(string), 0])
    print(string + ' ' * cntPadding, end = "\r", file = stream)
    printWithCr.previousStringLength = len(string)

printWithCr.previousStringLength = 0
