#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep 25 16:22:36 2018

@author: Nicolas Peteilh, François Gallard, Thales Delmiro
"""
import numpy as __np
import marilib
import sys

sys.modules["marilib"].numpy = __np
sys.modules["marilib.numpy"] = __np


def use_autograd():
    from autograd import numpy as __agnpy
    sys.modules["marilib"].numpy = __agnpy
    sys.modules["marilib.numpy"] = __agnpy
