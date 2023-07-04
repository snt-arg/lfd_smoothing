#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun 13 09:51:31 2023

@author: abrk
"""

import numpy as np
import matplotlib.pyplot as plt
from pydrake.all import *


num_control_points = 5
dof = 1
num_waypoints = 3 # Including start and goal
bspline_order = 4



#%%

trajectory = BsplineTrajectory(BsplineBasis(bspline_order,num_control_points,KnotVectorType.kClampedUniform,0,1),
                                 [np.zeros((dof, 1)) for _ in range(num_control_points)])

normalized_knots = trajectory.basis().knots()
basis = BsplineBasis_[Expression](trajectory.basis().order(), normalized_knots)
basis_normal = BsplineBasis(trajectory.basis().order(), normalized_knots)


prog = MathematicalProgram()

cp1 = prog.NewContinuousVariables(dof, num_control_points, "cp1")
cp2 = prog.NewContinuousVariables(dof, num_control_points, "cp2")

prog.SetInitialGuess(cp1, np.array(trajectory.control_points()).squeeze(-1).transpose())
prog.SetInitialGuess(cp2, np.array(trajectory.control_points()).squeeze(-1).transpose())


ufunc = np.vectorize(Expression)

cp1_exp = [ufunc(cp1[:,i,np.newaxis]) for i in range(num_control_points)]
cp2_exp = [ufunc(cp2[:,i,np.newaxis]) for i in range(num_control_points)]

sym_r1 = BsplineTrajectory_[Expression](basis,cp1_exp)
sym_r2 = BsplineTrajectory_[Expression](basis,cp2_exp)

sym_r1_dot = sym_r1.MakeDerivative()
sym_r2_dot = sym_r2.MakeDerivative()

sym_r1_ddot = sym_r1_dot.MakeDerivative()
sym_r2_ddot = sym_r2_dot.MakeDerivative()


# r1(0) = 0
prog.AddLinearConstraint(sym_r1.value(0), [2], [2])
# r1_dot(0) = 0
prog.AddLinearConstraint(sym_r1_dot.value(0), [1], [1])
# r1(1) = 1
prog.AddLinearConstraint(sym_r1.value(1), [1], [1])
# r1_dot(1) - r2_dot(0) = 0
prog.AddLinearConstraint(sym_r1_dot.value(1) - sym_r2_dot.value(0), [0], [0])
# r1_ddot(1) - r2_ddot(0) = 0
# prog.AddLinearConstraint(sym_r1_ddot.value(1) - sym_r2_ddot.value(0), [0], [0])
# r2(0) = 1
prog.AddLinearConstraint(sym_r2.value(0), [1], [1])
# r2(1) = 0
prog.AddLinearConstraint(sym_r2.value(1), [0], [0])
# r2_dot(1) = 0
prog.AddLinearConstraint(sym_r2_dot.value(1), [0], [0])

result = Solve(prog)

#%%

traj1 = BsplineTrajectory(basis_normal, [result.GetSolution(cp1)[:,i,np.newaxis] for i in range(num_control_points)])
traj2 = BsplineTrajectory(basis_normal, [result.GetSolution(cp2)[:,i,np.newaxis] for i in range(num_control_points)])

v_traj1 = traj1.MakeDerivative()
v_traj2 = traj2.MakeDerivative()


a_traj1 = v_traj1.MakeDerivative()
a_traj2 = v_traj2.MakeDerivative()


time = np.linspace(0, 1, 100)

t_tot = np.linspace(0, 2, 199)


# Plot position, velocity, and acceleration
fig, axs = plt.subplots(3, 1, figsize=(8, 10))


position = []
for t in time:
    position.append(traj1.value(t)[0])
    
position.pop()
    
for t in time:
    position.append(traj2.value(t)[0])

# Position plot
axs[0].plot(t_tot, position)
axs[0].set_ylabel('Position')
axs[0].set_xlabel('Time')

velocity = []
for t in time:
    velocity.append(v_traj1.value(t)[0])

velocity.pop()

for t in time:
    velocity.append(v_traj2.value(t)[0])
    
# Velocity plot
axs[1].plot(t_tot, velocity)
axs[1].set_ylabel('Velocity')
axs[1].set_xlabel('Time')

acceleration = []
for t in time:
    acceleration.append(a_traj1.value(t)[0])

acceleration.pop()

for t in time:
    acceleration.append(a_traj2.value(t)[0])

# Acceleration plot
axs[2].plot(t_tot, acceleration)
axs[2].set_ylabel('Acceleration')
axs[2].set_xlabel('Time')

plt.tight_layout()
plt.show()