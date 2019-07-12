#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep 25 16:22:36 2018

@author: Nicolas Peteilh, François Gallard, Thales Delmiro
"""
import numpy as __np
import marilib
import sys

marilib.is_using_autograd = False
sys.modules["marilib"].numpy = __np
sys.modules["marilib.numpy"] = __np


def use_autograd():
    from autograd import numpy as __agnpy
    sys.modules["marilib"].numpy = __agnpy
    sys.modules["marilib.numpy"] = __agnpy
    marilib.is_using_autograd = True
    from autograd import jacobian
    from autograd.numpy import array
    from numpy.linalg import solve as _np_solve
    from numpy import dot as _np_dot
    from scipy.optimize import fsolve as _scip_fsolve
    from autograd.extend import primitive, defvjp

    @primitive
    def fsolve(res, y, x, args=()):
        args = (x,) + args
        fprime = jacobian(res, argnum=0)
        return _scip_fsolve(res, y, x, fprime=fprime)

    @primitive
    def fsolve_vjp(g, ans, res, y0, x, args=()):
        drdy = jacobian(res, argnum=0)(ans, x, args)
        drdx = jacobian(res, argnum=1)(ans, x, args)
        dr_dx_g = _np_dot(drdx, g)
        dy_dx_dot = -_np_solve(drdy, dr_dx_g)
        return dy_dx_dot

    defvjp(fsolve, lambda ans, res, y0, x, args: lambda g: fsolve_vjp(g, ans, res, y0, x, args),
           argnums=[2])

    __agnpy.fsolve = fsolve
