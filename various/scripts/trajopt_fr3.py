#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun  7 10:37:37 2023

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
    RevoluteJoint
)

from manipulation import running_as_notebook
from manipulation.meshcat_utils import PublishPositionTrajectory
from manipulation.scenarios import AddIiwa, AddPlanarIiwa, AddShape, AddWsg
from manipulation.utils import ConfigureParser


#%%

meshcat = StartMeshcat()

#%%

def AddFrankaFr3(plant):
    urdf = "package://my_drake/manipulation/models/franka_description/urdf/fr3.urdf"

    parser = Parser(plant)
    parser.package_map().AddPackageXml("/home/abrk/catkin_ws/src/franka/franka_ros/franka_description/package.xml")
    parser.package_map().AddPackageXml("/home/abrk/thesis/drake/my_drake/package.xml")
    fr3 = parser.AddModelsFromUrl(urdf)[0]
    plant.WeldFrames(plant.world_frame(), plant.GetFrameByName("fr3_link0"))

    # Set default positions:
    q0 = [0.0, -0.78, 0.0, -2.36, 0.0, 1.57, 0.78, 0.035, 0.035]
    index = 0
    for joint_index in plant.GetJointIndices(fr3):
        joint = plant.get_mutable_joint(joint_index)
        if isinstance(joint, RevoluteJoint):
            joint.set_default_angle(q0[index])
            index += 1

    return fr3


def create_waypoint(name, x, y, z):
    X = RigidTransform([x, y, z])
    meshcat.SetObject(name, Sphere(0.02), rgba=Rgba(0.9, 0.1, 0.1, 1))
    meshcat.SetTransform(name, X)
    return X


#%%

meshcat.Delete()
builder = DiagramBuilder()

plant, scene_graph = AddMultibodyPlantSceneGraph(builder, time_step=0.001)
fr3 = AddFrankaFr3(plant)

waypoints = []

waypoints.append(create_waypoint("start", 0.5, 0.3, 0.2))
waypoints.append(create_waypoint("w1", 0.5, 0.2, 0.35))
waypoints.append(create_waypoint("w2", 0.5, 0.0, 0.35))
waypoints.append(create_waypoint("w3", 0.5, -0.2, 0.35))
waypoints.append(create_waypoint("goal", 0.5, -0.3, 0.2))


parser = Parser(plant)
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
gripper_frame = plant.GetFrameByName("fr3_hand_tcp", fr3)

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
    waypoints[0].translation(),
    waypoints[0].translation(),
    gripper_frame,
    [0, 0, 0],
    plant_context,
)
trajopt.AddPathPositionConstraint(start_constraint, 0)
prog.AddQuadraticErrorCost(
    np.eye(num_q), q0, trajopt.control_points()[:, 0]
)

# trajopt.AddPathAccelerationConstraint(np.array([[0,0,1]]).transpose(),np.array([[0,0,10]]).transpose(),0)

# middle constraint
middle_constraint = PositionConstraint(
    plant,
    plant.world_frame(),
    waypoints[1].translation() - [0.05,0.05,0.05],
    waypoints[1].translation() + [0.05,0.05,0.05],
    gripper_frame,
    [0, 0, 0],
    plant_context,
)
trajopt.AddPathPositionConstraint(middle_constraint, 0.3)


middle_constraint2 = PositionConstraint(
    plant,
    plant.world_frame(),
    waypoints[2].translation() - [0.05,0.05,0.05],
    waypoints[2].translation() + [0.05,0.05,0.05],
    gripper_frame,
    [0, 0, 0],
    plant_context,
)
trajopt.AddPathPositionConstraint(middle_constraint2, 0.5)

middle_constraint3 = PositionConstraint(
    plant,
    plant.world_frame(),
    waypoints[3].translation() - [0.05,0.05,0.05],
    waypoints[3].translation() + [0.05,0.05,0.05],
    gripper_frame,
    [0, 0, 0],
    plant_context,
)
trajopt.AddPathPositionConstraint(middle_constraint3, 0.7)




# goal constraint
goal_constraint = PositionConstraint(
    plant,
    plant.world_frame(),
    waypoints[-1].translation(),
    waypoints[-1].translation(),
    gripper_frame,
    [0, 0, 0],
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
    
trajopt.SetInitialGuess(trajopt.ReconstructTrajectory(result))

result = Solve(prog)
if not result.is_success():
    print("Trajectory optimization failed")
    print(result.get_solver_id().name())
    
monitor_var = np.array(trajopt.control_points())

PublishPositionTrajectory(
    trajopt.ReconstructTrajectory(result), context, plant, visualizer
)
collision_visualizer.ForcedPublish(
    collision_visualizer.GetMyContextFromRoot(context)
)


#%% Playground

from pydrake.all import (MathematicalProgram)

consts = prog.GetAllConstraints()

nprog = MathematicalProgram()

nprog.AddDecisionVariables(prog.decision_variables())

for const in consts:
    nprog.AddConstraint(const.evaluator(), const.variables())
    
costs = prog.GetAllCosts()

for cost in costs:
    nprog.AddCost(cost.evaluator(), cost.variables())


result = Solve(nprog)

result.is_success()


# consts[0].variables()[0].get_id()


# points = trajopt.control_points()
