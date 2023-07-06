#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun  8 16:37:45 2023

@author: abrk
"""

#%%
from math import inf
import numpy as np
from scipy.interpolate import CubicSpline
import matplotlib.pyplot as plt
import pickle
from scipy.interpolate import interp1d

from pydrake.all import *


# with open('0.pickle', 'rb') as file:
#     demonstration = pickle.load(file)
  
# joint_trajectory = demonstration.joint_trajectory

# n_time_steps = len(joint_trajectory.points)
# n_dim = len(joint_trajectory.joint_names)

# ys = np.zeros([n_time_steps,n_dim])
# ts = np.zeros(n_time_steps)
# ts = ts / ts[-1]

# for (i,point) in enumerate(joint_trajectory.points):
#     ys[i,:] = point.positions
#     ts[i] = point.time_from_start.to_sec()


# pose_trajectory = demonstration.pose_trajectory

# n_time_steps = len(joint_trajectory.points)

# positions = np.zeros((n_time_steps, 3))
# orientations = np.zeros((n_time_steps, 4))

# for (i,point) in enumerate(pose_trajectory.points):
#     positions[i,:] = [point.pose.position.x, point.pose.position.y, point.pose.position.z]
#     orientations[i,:] = [point.pose.orientation.x, point.pose.orientation.y, point.pose.orientation.z, point.pose.orientation.w] 
    
#%%
# duration = 1
num_positions = 3
num_control_points = 10
bspline_order = 4

trajectory = BsplineTrajectory(BsplineBasis(bspline_order,num_control_points,KnotVectorType.kClampedUniform,0,1),
                                 [np.zeros((num_positions, 1)) for _ in range(num_control_points)])

normalized_knots = trajectory.basis().knots()
basis = BsplineBasis_[Expression](trajectory.basis().order(), normalized_knots)
basis_normal = BsplineBasis(trajectory.basis().order(), normalized_knots)

prog = MathematicalProgram()

control_points = prog.NewContinuousVariables(num_positions, num_control_points, "control_points")
duration = prog.NewContinuousVariables(1, "duration")[0]

prog.AddBoundingBoxConstraint(0, inf, duration)

prog.SetInitialGuess(control_points, np.array(trajectory.control_points()).squeeze(-1).transpose())
prog.SetInitialGuess(duration, trajectory.end_time() - trajectory.start_time())

ufunc = np.vectorize(Expression)
expression = [ufunc(control_points[:,i,np.newaxis]) for i in range(num_control_points)]

sym_r = BsplineTrajectory_[Expression](basis,expression)

sym_rdot = sym_r.MakeDerivative()
sym_rddot = sym_rdot.MakeDerivative()



#%%

# for i in range(positions.shape[0]):
#     prog.AddLinearConstraint(sym_r.value(ts[i]), positions[i,:], positions[i,:])


# Start constraint
sym_r_value = sym_r.value(0)
prog.AddLinearConstraint(sym_r_value, [1,1,1], [1,1,1])

sym_rdot_value = sym_rdot.value(0)
prog.AddLinearConstraint(sym_rdot_value, [0,0,0], [0,0,0])


# Goal Constraint
sym_r_value = sym_r.value(1)
prog.AddLinearConstraint(sym_r_value, [4,5,6], [4,5,6])

sym_rdot_value = sym_rdot.value(1)
prog.AddLinearConstraint(sym_rdot_value, [0,0,0], [0,0,0])

#%%


#%%

result = Solve(prog)

resulting_traj = BsplineTrajectory(basis_normal, [result.GetSolution(control_points)[:,i,np.newaxis] for i in range(num_control_points)])
v_traj = resulting_traj.MakeDerivative()
a_traj = v_traj.MakeDerivative()

time = np.linspace(0, 1, 100)


# Plot position, velocity, and acceleration
fig, axs = plt.subplots(3, 1, figsize=(8, 10))

position = []
for t in time:
    position.append(resulting_traj.value(t)[0])

# Position plot
axs[0].plot(time, position)
axs[0].set_ylabel('Position')
axs[0].set_xlabel('Time')

velocity = []
for t in time:
    velocity.append(v_traj.value(t)[0])
    
# Velocity plot
axs[1].plot(time, velocity)
axs[1].set_ylabel('Velocity')
axs[1].set_xlabel('Time')

acceleration = []
for t in time:
    acceleration.append(a_traj.value(t)[0])

# Acceleration plot
axs[2].plot(time, acceleration)
axs[2].set_ylabel('Acceleration')
axs[2].set_xlabel('Time')

plt.tight_layout()
plt.show()

# resulting_traj.value(1)
# resulting_traj.MakeDerivative().value(1)



















