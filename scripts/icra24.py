#!/usr/bin/env python3

import os
import rospy
import numpy as np
import matplotlib.pyplot as plt

from std_msgs.msg import Float64, Float64MultiArray
from sensor_msgs.msg import JointState

from pydrake.all import *

from lfd_smoother.util.icra import TrajectoryStock, TorqueAnalysis, CartesianAnalysis, ToleranceAnalysis

    # f = FrequencyAnalysis()
    # f.run(demo_name)

    # torque_analysis = TorqueAnalysis()
    # print(torque_analysis.run(original_traj))


    # cartesian_analyser = CartesianAnalysis()
    # cartesian_analyser.from_trajectory(smooth_traj)
    # cartesian_analyser.plot()


def tolerance_analysis():
    smooth_traj = TrajectoryStock()
    smooth_traj.import_from_pydrake("smooth"+demo_name, t_scale=0)

    correct_traj = TrajectoryStock()
    correct_traj.import_from_pydrake("correct"+demo_name, t_scale=0)

    cartesian_analyser = CartesianAnalysis()
    cartesian_analyser.from_trajectory(correct_traj)

    tol_an = ToleranceAnalysis()
    tol_an.import_data("smooth" + demo_name)
    tol_an.plot_traj_with_tols(correct_traj)
  

if __name__ == '__main__':

    rospy.init_node('icra24')
    os.chdir(rospy.get_param("~working_dir"))

    demo_name = "picknplace0"

    tolerance_analysis()



    rospy.spin()