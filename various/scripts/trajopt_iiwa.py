#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun  7 11:32:12 2023

@author: abrk
"""

#%%
import time

import numpy as np
from pydrake.all import (
    AddMultibodyPlantSceneGraph,
    BsplineTrajectory,
    DiagramBuilder,
    KinematicTrajectoryOptimization,
    MeshcatVisualizer,
    MeshcatVisualizerParams,
    MinimumDistanceConstraint,
    Parser,
    PositionConstraint,
    Rgba,
    RigidTransform,
    Role,
    Solve,
    Sphere,
    StartMeshcat,
)

from manipulation import running_as_notebook
from manipulation.meshcat_utils import PublishPositionTrajectory
from manipulation.scenarios import AddIiwa, AddPlanarIiwa, AddShape, AddWsg
from manipulation.utils import ConfigureParser

#%%
# Start the visualizer.
meshcat = StartMeshcat()


#%%

meshcat.Delete()
builder = DiagramBuilder()

plant, scene_graph = AddMultibodyPlantSceneGraph(builder, time_step=0.001)
iiwa = AddPlanarIiwa(plant)
wsg = AddWsg(plant, iiwa, roll=0.0, welded=True, sphere=True)
X_WStart = RigidTransform([0.8, 0, 0.65])
meshcat.SetObject("start", Sphere(0.02), rgba=Rgba(0.9, 0.1, 0.1, 1))
meshcat.SetTransform("start", X_WStart)
X_WGoal = RigidTransform([0.8, 0, 0.4])
meshcat.SetObject("goal", Sphere(0.02), rgba=Rgba(0.1, 0.9, 0.1, 1))
meshcat.SetTransform("goal", X_WGoal)

parser = Parser(plant)
ConfigureParser(parser)
bin = parser.AddModelsFromUrl("package://manipulation/shelves.sdf")[0]
plant.WeldFrames(
    plant.world_frame(),
    plant.GetFrameByName("shelves_body", bin),
    RigidTransform([0.88, 0, 0.4]),
)

plant.Finalize()

visualizer = MeshcatVisualizer.AddToBuilder(
    builder,
    scene_graph,
    meshcat,
    MeshcatVisualizerParams(role=Role.kIllustration),
)
collision_visualizer = MeshcatVisualizer.AddToBuilder(
    builder,
    scene_graph,
    meshcat,
    MeshcatVisualizerParams(prefix="collision", role=Role.kProximity),
)
meshcat.SetProperty("collision", "visible", False)

diagram = builder.Build()
context = diagram.CreateDefaultContext()
plant_context = plant.GetMyContextFromRoot(context)

num_q = plant.num_positions()
q0 = plant.GetPositions(plant_context)
gripper_frame = plant.GetFrameByName("body", wsg)

trajopt = KinematicTrajectoryOptimization(plant.num_positions(), 10)
prog = trajopt.get_mutable_prog()
trajopt.AddDurationCost(1.0)
trajopt.AddPathLengthCost(1.0)
trajopt.AddPositionBounds(
    plant.GetPositionLowerLimits(), plant.GetPositionUpperLimits()
)
trajopt.AddVelocityBounds(
    plant.GetVelocityLowerLimits(), plant.GetVelocityUpperLimits()
)

trajopt.AddDurationConstraint(0.5, 5)

# start constraint
start_constraint = PositionConstraint(
    plant,
    plant.world_frame(),
    X_WStart.translation(),
    X_WStart.translation(),
    gripper_frame,
    [0, 0.1, 0],
    plant_context,
)
trajopt.AddPathPositionConstraint(start_constraint, 0)
prog.AddQuadraticErrorCost(
    np.eye(num_q), q0, trajopt.control_points()[:, 0]
)

# goal constraint
goal_constraint = PositionConstraint(
    plant,
    plant.world_frame(),
    X_WGoal.translation(),
    X_WGoal.translation(),
    gripper_frame,
    [0, 0.1, 0],
    plant_context,
)
trajopt.AddPathPositionConstraint(goal_constraint, 1)
prog.AddQuadraticErrorCost(
    np.eye(num_q), q0, trajopt.control_points()[:, -1]
)

# start and end with zero velocity
trajopt.AddPathVelocityConstraint(
    np.zeros((num_q, 1)), np.zeros((num_q, 1)), 0
)
trajopt.AddPathVelocityConstraint(
    np.zeros((num_q, 1)), np.zeros((num_q, 1)), 1
)

# Solve once without the collisions and set that as the initial guess for
# the version with collisions.
result = Solve(prog)
if not result.is_success():
    print("Trajectory optimization failed, even without collisions!")
    print(result.get_solver_id().name())


# trajopt.SetInitialGuess(trajopt.ReconstructTrajectory(result))

# # collision constraints
# collision_constraint = MinimumDistanceConstraint(
#     plant, 0.001, plant_context, None, 0.01
# )
# evaluate_at_s = np.linspace(0, 1, 25)
# for s in evaluate_at_s:
#     trajopt.AddPathPositionConstraint(collision_constraint, s)

# def PlotPath(control_points):
#     traj = BsplineTrajectory(
#         trajopt.basis(), control_points.reshape((3, -1))
#     )
#     meshcat.SetLine(
#         "positions_path", traj.vector_values(np.linspace(0, 1, 50))
#     )

# prog.AddVisualizationCallback(
#     PlotPath, trajopt.control_points().reshape((-1,))
# )
# result = Solve(prog)
# if not result.is_success():
#     print("Trajectory optimization failed")
#     print(result.get_solver_id().name())

PublishPositionTrajectory(
    trajopt.ReconstructTrajectory(result), context, plant, visualizer
)
collision_visualizer.ForcedPublish(
    collision_visualizer.GetMyContextFromRoot(context)
)
