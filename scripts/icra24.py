#!/usr/bin/env python3

import os
import rospy
import numpy as np
import matplotlib.pyplot as plt

from std_msgs.msg import Float64, Float64MultiArray
from sensor_msgs.msg import JointState

from pydrake.all import *

from lfd_smoother.util.icra import TrajectoryStock, TorqueAnalysis, CartesianAnalysis, ToleranceAnalysis, JerkAnalysis

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

    smooth_cart = CartesianAnalysis()
    smooth_cart.from_trajectory(smooth_traj)

    correct_cart = CartesianAnalysis()
    correct_cart.from_trajectory(correct_traj)

    tol_an = ToleranceAnalysis()
    tol_an.import_data("smooth" + demo_name)
    tol_an.plot_traj_with_tols(correct_traj, smooth_traj)
    return tol_an

def jerk_analysis():
    original_traj = TrajectoryStock()
    original_traj.import_from_lfd_storage("filter"+demo_name, t_scale=0)

    original_traj_n = TrajectoryStock()
    original_traj_n.import_from_lfd_storage("filter"+demo_name, t_scale=1)

    smooth_traj = TrajectoryStock()
    smooth_traj.import_from_pydrake("smooth"+demo_name, t_scale=0)

    smooth_traj_n = TrajectoryStock()
    smooth_traj_n.import_from_pydrake("smooth"+demo_name, t_scale=1)

    jerk_analysis = JerkAnalysis()
    jerk_analysis.plot_with_high_jerk(original_traj.ts, original_traj.ys[:,1], original_traj_n.yddds[:,1])
    jerk_analysis.plot_with_low_jerk(smooth_traj.ts, smooth_traj.ys[:,1], smooth_traj_n.yddds[:,1])

def velocity_adjustment_analysis():
    smooth_traj = TrajectoryStock()
    smooth_traj.import_from_pydrake("smooth"+demo_name, t_scale=0)

    refined_traj = TrajectoryStock()
    refined_traj.import_from_pydrake("correct"+demo_name, t_scale=0)

    smooth_cart = CartesianAnalysis()
    smooth_cart.from_trajectory(smooth_traj)

    refined_cart = CartesianAnalysis()
    refined_cart.from_trajectory(refined_traj)

    # smooth_cart.plot_3d()
    # refined_cart.plot_3d()

    smooth_traj.plot()
    refined_traj.plot()    

if __name__ == '__main__':

    rospy.init_node('icra24')
    os.chdir(rospy.get_param("~working_dir"))

    demo_name = "picknplace0"

    # original_traj = TrajectoryStock()
    # original_traj.import_from_lfd_storage("filter"+demo_name, t_scale=0)   
    # original_traj.plot() 

    tol_an = tolerance_analysis()
    # velocity_adjustment_analysis()


    # rospy.spin()