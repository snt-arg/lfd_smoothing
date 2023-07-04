#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun 13 14:49:47 2023

@author: abrk
"""

import numpy as np
import matplotlib.pyplot as plt
from pydrake.all import *


num_control_points = 10
dof = 1
num_waypoints = 3 # Including start and goal
bspline_order = 10

#%%

trajopts = []
progs = []

trajopts.append(KinematicTrajectoryOptimization(dof, num_control_points, spline_order=bspline_order))
trajopts.append(KinematicTrajectoryOptimization(dof, num_control_points, spline_order=bspline_order))
trajopts.append(KinematicTrajectoryOptimization(dof, num_control_points, spline_order=bspline_order))

for trajopt in trajopts:
    progs.append(trajopt.get_mutable_prog())
    trajopt.AddDurationCost(1.0)
    trajopt.AddPathLengthCost(1)
    trajopt.AddPositionBounds([-5], [5])
    trajopt.AddVelocityBounds([-0.5], [0.5])
    trajopt.AddDurationConstraint(0.1, 5)

trajopts[0].AddPathPositionConstraint([-0.1], [0.1], 0) #
trajopts[0].AddPathVelocityConstraint([0], [0], 0) #
trajopts[0].AddPathPositionConstraint([1], [1], 1)

trajopts[1].AddPathPositionConstraint([1], [1], 0)
trajopts[1].AddPathPositionConstraint([1.8], [2.2], 1)

trajopts[2].AddPathPositionConstraint([1.8], [2.2], 0)
trajopts[2].AddPathPositionConstraint([3], [3], 1) #
trajopts[2].AddPathVelocityConstraint([0], [0], 1) #

#%%

nprog = MathematicalProgram()

for prog in progs:
    nprog.AddDecisionVariables(prog.decision_variables())
    for const in prog.GetAllConstraints():
        nprog.AddConstraint(const.evaluator(), const.variables())
    for cost in prog.GetAllCosts():
        nprog.AddCost(cost.evaluator(), cost.variables())

#%%

def create_sym_r(trajopt):
    control_points = trajopt.control_points()
    ufunc = np.vectorize(Expression)
    control_points_exp = [ufunc(control_points[:,i,np.newaxis]) for i in range(num_control_points)]
    basis_exp = BsplineBasis_[Expression](trajopt.basis().order(), trajopt.basis().knots())
    
    sym_r = BsplineTrajectory_[Expression](basis_exp,control_points_exp)
    # sym_rdot = sym_r.MakeDerivative()
    # sym_rddot = sym_rdot.MakeDerivative()
    
    return sym_r #, sym_rdot, sym_rddot

sym_rs = []

for trajopt in trajopts:
    sym_rs.append(create_sym_r(trajopt))
    
# (sym_rs[0].MakeDerivative().value(1) * trajopts[0].duration()) - (sym_rs[1].MakeDerivative().value(0) * trajopts[1].duration())
nprog.AddConstraint(((sym_rs[0].MakeDerivative().value(1) / trajopts[0].duration()) 
                    - (sym_rs[1].MakeDerivative().value(0) / trajopts[1].duration()))[0][0]
                    , 0, 0)
nprog.AddConstraint(((sym_rs[1].MakeDerivative().value(1) / trajopts[1].duration()) 
                    - (sym_rs[2].MakeDerivative().value(0) / trajopts[2].duration()))[0][0]
                    , 0, 0)

nprog.AddConstraint(((sym_rs[0].MakeDerivative(2).value(1) / trajopts[0].duration() / trajopts[0].duration()) 
                    - (sym_rs[1].MakeDerivative(2).value(0) / trajopts[1].duration() / trajopts[1].duration()))[0][0]
                    , 0, 0)
nprog.AddConstraint(((sym_rs[1].MakeDerivative(2).value(1) / trajopts[1].duration()/ trajopts[1].duration()) 
                    - (sym_rs[2].MakeDerivative(2).value(0) / trajopts[2].duration()/trajopts[2].duration()))[0][0]
                    , 0, 0)
nprog.AddLinearConstraint(sym_rs[0].value(1) - sym_rs[1].value(0), [0], [0])
nprog.AddLinearConstraint(sym_rs[1].value(1) - sym_rs[2].value(0), [0], [0])

#%%
def reconstruct_trajectory(result, trajopt, t_offset):
    duration = result.GetSolution(trajopt.duration())
    basis = trajopt.basis()
    scaled_knots = [(knot * duration) + t_offset for knot in basis.knots()]
    
    return duration, BsplineTrajectory(
        BsplineBasis(basis.order(), scaled_knots),
        [result.GetSolution(trajopt.control_points())[:,i,np.newaxis] for i in range(num_control_points)])

#%%
# knots_total = []
# cps_total = []

# for trajopt in trajopts:
#     duration = result.GetSolution(trajopt.duration())
#     basis = trajopt.basis()
#     scaled_knots = [knot * duration for knot in basis.knots()]
#     cps = [result.GetSolution(trajopt.control_points())[:,i,np.newaxis] for i in range(num_control_points)]
    
#     knots_total.extend(scaled_knots)
#     cps_total.extend(cps)
    
# combined_traj = BsplineTrajectory(
#     BsplineBasis(bspline_order, knots_total),
#     cps_total)


#%%

result = Solve(nprog)

result

if not result.is_success():
    print("optimization failed")

output_trajs = []
t_offset = 0.0
for trajopt in trajopts:
    duration, traj = reconstruct_trajectory(result, trajopt, t_offset)
    output_trajs.append(traj)
    t_offset += duration
    
#%%


composite_traj = CompositeTrajectory(output_trajs)

    
#%%
ts = np.linspace(0, composite_traj.end_time(), 1000)

position = []
velocity = []
acceleration = []
jerk = []

for t in ts:
    position.append(composite_traj.value(t)[0])
    velocity.append(composite_traj.MakeDerivative().value(t)[0])
    acceleration.append(composite_traj.MakeDerivative(2).value(t)[0])
    jerk.append(composite_traj.MakeDerivative(3).value(t)[0])

# Plot position, velocity, and acceleration
fig, axs = plt.subplots(4, 1, figsize=(8, 10))

# Position plot
axs[0].plot(ts, position)
axs[0].set_ylabel('Position')
axs[0].set_xlabel('Time')

# Velocity plot
axs[1].plot(ts, velocity)
axs[1].set_ylabel('Velocity')
axs[1].set_xlabel('Time')

# Acceleration plot
axs[2].plot(ts, acceleration)
axs[2].set_ylabel('Acceleration')
axs[2].set_xlabel('Time')

# Jerk plot
axs[3].plot(ts, jerk)
axs[3].set_ylabel('Jerk')
axs[3].set_xlabel('Time')

plt.tight_layout()
plt.show()

#%%


# t_total = 0.0

# for traj in output_trajs:
#     t_total = t_total + traj.end_time()

# ts = np.linspace(0, t_total, 1000)

# position = []
# velocity = []
# acceleration = []
# jerk = []

# for t in ts:
#     if t < output_trajs[0].end_time():
#         position.append(output_trajs[0].value(t)[0])
#         velocity.append(output_trajs[0].MakeDerivative().value(t)[0])
#         acceleration.append(output_trajs[0].MakeDerivative(2).value(t)[0])
#         jerk.append(output_trajs[0].MakeDerivative(3).value(t)[0])
#     else:
#         position.append(output_trajs[1].value(t)[0])
#         velocity.append(output_trajs[1].MakeDerivative().value(t)[0])
#         acceleration.append(output_trajs[1].MakeDerivative(2).value(t)[0])
#         jerk.append(output_trajs[1].MakeDerivative(2).value(t)[0])
        
# # Plot position, velocity, and acceleration
# fig, axs = plt.subplots(4, 1, figsize=(8, 10))

# # Position plot
# axs[0].plot(ts, position)
# axs[0].set_ylabel('Position')
# axs[0].set_xlabel('Time')

# # Velocity plot
# axs[1].plot(ts, velocity)
# axs[1].set_ylabel('Velocity')
# axs[1].set_xlabel('Time')

# # Acceleration plot
# axs[2].plot(ts, acceleration)
# axs[2].set_ylabel('Acceleration')
# axs[2].set_xlabel('Time')

# # Jerk plot
# axs[3].plot(ts, jerk)
# axs[3].set_ylabel('Jerk')
# axs[3].set_xlabel('Time')

# plt.tight_layout()
# plt.show()