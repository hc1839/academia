r"""
Facilities for easier calling of Sympy expressions as functions.
"""

import inspect
import sympy

class SympyFn(object):
    r"""
    Wrapper that keeps a Sympy expression and its lambdified form together.

    **Data members:**
        ``_sympyParams``: Tuple of parameter symbols. See ``sympyParams`` in
        :func:`__init__`.

        ``_expr``: Sympy expression.

        ``_exprAsFn``: Lambidified form of the Sympy expression.

        ``_module``: See :func:`module`.
    """

    def __init__(self, sympyParams, sympyExpr, module):
        r"""
        :param sympyParams:
            Sequence of symbols that are parameters of the expression.

        :param sympyExpr:
            Sympy expression.

        :param module:
            Module to use for Sympy's lambdify.
        """
        sympyParams = tuple(sympyParams)

        if (not(inspect.ismodule(module))):
            raise TypeError()

        self._sympyParams = sympyParams
        self._expr = sympyExpr
        self._exprAsFn = sympy.lambdify(sympyParams, sympyExpr, module)
        self._module = module

    def __call__(self, *argv):
        r"""
        Call the Sympy expression as a function with the given arguments.

        :param argv:
            Argument vector for the lambidifed form of the Sympy expression.

        :return:
            Value of the Sympy expression evaluated with the given arguments.
        """
        return self._exprAsFn(*argv)

    def parameters(self):
        r"""
        Parameters of the expression.

        :return:
            Parameters as Sympy symbols.

        :rtype: tuple
        """
        return self._sympyParams

    def expression(self):
        r"""
        Sympy expression.

        :return:
            Sympy expression.
        """
        return self._expr

    def module(self):
        r"""
        Module used for Sympy's lambdify.

        :return:
            Module used for Sympy's lambdify.

        :rtype: module
        """
        return self._module
