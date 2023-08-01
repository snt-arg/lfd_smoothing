#!/usr/bin/env python3

import os
import rospy
import numpy as np
import matplotlib.pyplot as plt

from std_msgs.msg import Float64, Float64MultiArray
from sensor_msgs.msg import JointState

from pydrake.all import *

from lfd_smoother.util.icra import TrajectoryStock, TorqueAnalysis
from lfd_smoother.util.fr3_drake import FR3Drake


class CartesianAnalysis:

    def __init__(self) -> None:
        self.robot = FR3Drake(franka_pkg_xml="/home/abrk/catkin_ws/src/lfd/lfd_smoothing/drake/franka_drake/package.xml",
                urdf_path="package://franka_drake/urdf/fr3_nohand.urdf")
        self.visualizer_context = self.robot.visualizer.GetMyContextFromRoot(self.robot.context)
        self.ee_body = self.robot.plant.GetBodyByName("fr3_link8")

    def to_position(self, q):
        self.robot.plant.SetPositions(self.robot.plant_context, q)
        transform = self.robot.plant.EvalBodyPoseInWorld(self.robot.plant_context, self.ee_body)
        return transform.translation()
    
    def to_velocity(self, q, qd):
        qqd = np.concatenate((q,qd), axis=0)
        self.robot.plant.SetPositionsAndVelocities(self.robot.plant_context, qqd)
        velocity = self.robot.plant.EvalBodySpatialVelocityInWorld(self.robot.plant_context, self.ee_body)
        return np.linalg.norm(velocity.translational())




if __name__ == '__main__':

    rospy.init_node('icra24')
    os.chdir(rospy.get_param("~working_dir"))

    demo_name = "picknplace0"


    # f = FrequencyAnalysis()
    # f.run(demo_name)

    # torque_analysis = TorqueAnalysis()
    # print(torque_analysis.run(original_traj))

    smooth_traj = TrajectoryStock()
    smooth_traj.import_from_pydrake("smooth"+demo_name, t_scale=0)

    # original_traj = TrajectoryStock()
    # original_traj.import_from_lfd_storage("filter"+demo_name, t_scale=2)


    cartesian_analyser = CartesianAnalysis()
    transform = cartesian_analyser.to_position(smooth_traj.value(1))

    print(cartesian_analyser.to_velocity(smooth_traj.value(1), np.zeros((7,1))))



    # robot = FR3Drake(franka_pkg_xml="/home/abrk/catkin_ws/src/lfd/lfd_smoothing/drake/franka_drake/package.xml",
    #                 urdf_path="package://franka_drake/urdf/fr3_nohand.urdf")

    # visualizer_context = robot.visualizer.GetMyContextFromRoot(robot.context)

    # robot.plant.SetPositions(robot.plant_context, smooth_traj.value(1))
    # transform = robot.plant.EvalBodyPoseInWorld(robot.plant_context, robot.plant.GetBodyByName("fr3_link8"))
    # cartesian_analyser.robot.meshcat.SetObject("dada", Sphere(0.003), rgba=Rgba(0.9, 0.1, 0.1, 1))
    # cartesian_analyser.robot.meshcat.SetTransform("dada", transform)
    cartesian_analyser.robot.visualizer.ForcedPublish(cartesian_analyser.visualizer_context)

    # robot.plant_context.SetDiscreteState(np.array([0 for _ in range(14)]))
    # print(robot.plant_context)

    rospy.spin()