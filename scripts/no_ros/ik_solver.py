#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct  5 17:25:28 2023

@author: abrk
"""

import numpy as np

from pydrake.all import *

from lfd_smoother.util.robot import YumiDrake


#%%

config= {
"demo_filter_threshold": 0.01,
"pkg_xml": "drake/yumi_drake/package.xml",
"urdf_path": "package://yumi_drake/urdf/yumi_right.urdf",
}

robot = YumiDrake(pkg_xml=config["pkg_xml"],
                        urdf_path=config["urdf_path"], left_arm=False)

pose = robot.create_waypoint("pose", [0.303, 0, 0.482])
pose.set_rotation(Quaternion([0,-1,0,0]))
q0 = np.zeros(robot.plant.num_positions())
# q0 = [0.00112954, -0.787257, -0.000108279, -2.36501, 0.00214555, 1.55991, 0.766295,1,1]

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

#%%

robot.visualize(q)
