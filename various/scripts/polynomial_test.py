#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun 12 13:43:42 2023

@author: abrk
"""

import numpy as np
import matplotlib.pyplot as plt

from pydrake.all import *

prog = MathematicalProgram()

cs= prog.NewContinuousVariables(2,"c")
ts = prog.NewContinuousVariables(2, "time")

def func(c,t):
    return c[0] * (t * t) + c[1]

prog.AddCost((func(cs, ts[0]) - 3) ** 2 + (func(cs, ts[1]) - 9) ** 2)
prog.AddConstraint(ts[0] <= ts[1])
prog.AddConstraint(ts[1] - ts[0] ==1)
prog.AddConstraint(ts[1]**2 - ts[0]**2 == 3)


prog.AddBoundingBoxConstraint(np.array([0,0]), np.array([3,3]), cs)
prog.AddBoundingBoxConstraint(np.array([0,0]), np.array([3,3]), ts)

cs_init = np.array([1.5,0.5])
ts_init = np.array([1.5,3])

# prog.SetInitialGuess(cs, cs_init)
# prog.SetInitialGuess(ts, ts_init)

result = Solve(prog)

result.is_success()

#%%
c_real = [2,1]
ts_real = [1,2]

cs_result = result.GetSolution(cs)
ts_result = result.GetSolution(ts)

cs_result - c_real

ts_result[1]**2 - ts_result[0]**2

(func(cs_result, ts_result[0]) - 3) ** 2 + (func(cs_result, ts_result[1]) - 9) ** 2

