#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct  5 17:25:28 2023

@author: abrk
"""

import numpy as np

from pydrake.all import *

from lfd_smoother.core.trajectory_optimizer import TrajectoryOptimizer
from lfd_smoother.core.single_optimizer import SingleOptimizer
from lfd_smoother.util.demonstration import Demonstration
from lfd_smoother.util.fr3_drake import FR3Drake
from lfd_smoother.util.config import FR3Config

#%%

config= {
"demo_filter_threshold": 0.01,
"franka_pkg_xml": "drake/franka_drake/package.xml",
"urdf_path": "package://franka_drake/urdf/fr3_full.urdf",
"config_hierarchy": ["config/opt/initial.yaml", "config/opt/main.yaml"]
}

robot = FR3Drake(franka_pkg_xml=config["franka_pkg_xml"],
                        urdf_path=config["urdf_path"],
                        gripper_frame="fr3_hand_tcp")

pose = robot.create_waypoint("pose", [0.303, 0, 0.482])
pose.set_rotation(Quaternion([0,-1,0,0]))
q0 = np.zeros(robot.plant.num_positions())
q0 = [0.00112954, -0.787257, -0.000108279, -2.36501, 0.00214555, 1.55991, 0.766295,1,1]

#%%

ik_solver = InverseKinematics(robot.plant, True)

ik_solver.AddPositionConstraint(robot.plant.world_frame(), pose.translation()
                                ,robot.gripper_frame, [0, 0, 0], [0, 0, 0])

ik_solver.AddOrientationConstraint(robot.gripper_frame, RotationMatrix(), robot.plant.world_frame(), pose.rotation(), 0)

prog = ik_solver.get_mutable_prog()
prog.AddCost((ik_solver.q()-q0).dot(ik_solver.q()-q0))
# prog.AddQuadraticErrorCost(0.01*np.eye(9), q_desired, ik_solver.q())
#%%


#%%

result = Solve(prog)
q = result.GetSolution(ik_solver.q())

plant_context = robot.plant.GetMyContextFromRoot(robot.context)
visualizer_context = robot.visualizer.GetMyContextFromRoot(robot.context)
robot.plant.SetPositions(plant_context, q)
robot.visualizer.ForcedPublish(visualizer_context)