#!/usr/bin/env python3

import os
import rospy
import numpy as np
import matplotlib.pyplot as plt

from std_msgs.msg import Float64, Float64MultiArray
from sensor_msgs.msg import JointState

from pydrake.all import *

from lfd_smoother.util.icra import TrajectoryStock, TorqueAnalysis, CartesianAnalysis, ToleranceAnalysis, JerkAnalysis, DMPAnalysis

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
    # tol_an.plot_traj_with_tols(correct_traj)
    # tol_an.plot_x_with_tols(correct_traj)
    # tol_an.plot_t_s_command()
    tol_an.plot_waypoints(smooth_traj.ts[-1], correct_traj.ts[-1])
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

def tolerance_analysis2():
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
    tol_an.plot_normalized(smooth_traj, correct_traj)
    return tol_an

def dmp_analysis():
    original_dmp = TrajectoryStock()
    original_dmp.import_from_dmp_joint_trajectory("filter"+demo_name, t_scale=1)
    original_cart = CartesianAnalysis()
    original_cart.from_trajectory(original_dmp)

    scaled_dmp = TrajectoryStock()
    scaled_dmp.import_from_dmp_joint_trajectory("scaled"+demo_name, t_scale=0)
    scaled_cart = CartesianAnalysis()
    scaled_cart.from_trajectory(scaled_dmp)

    smooth_dmp = TrajectoryStock()
    smooth_dmp.import_from_dmp_joint_trajectory("smooth"+demo_name, t_scale=0)
    smooth_cart = CartesianAnalysis()
    smooth_cart.from_trajectory(smooth_dmp)

    traj = TrajectoryStock()
    traj.import_from_lfd_storage("filter"+demo_name, t_scale=0)
    traj_cart = CartesianAnalysis()
    traj_cart.from_trajectory(traj)

    dmp_an = DMPAnalysis()
    # dmp_an.plot_3d(original_dmp, smooth_dmp)
    # dmp_an.plot_abs_jerk(original_dmp, smooth_dmp)
    # dmp_an.plot_abs_jerk(original_dmp, smooth_dmp, scaled_dmp)
    
    # dmp_an.plot_with_kin_lims(scaled_dmp)
    # dmp_an.plot_with_kin_lims(smooth_dmp)
    # dmp_an.plot_vel_acc_kin_lims(scaled_dmp, smooth_dmp)
    dmp_an.plot_compare_jerks(scaled_dmp,smooth_dmp)

def traj_analysis():
    def max_abs_jerk(traj:TrajectoryStock):
        return np.max(np.abs(traj.yddds))
    
    def duration(traj:TrajectoryStock):
        return traj.ts[-1]
    
    original_dmp = TrajectoryStock()
    original_dmp.import_from_dmp_joint_trajectory("filter"+demo_name, t_scale=1)
    print(max_abs_jerk(original_dmp))


    scaled_dmp = TrajectoryStock()
    scaled_dmp.import_from_dmp_joint_trajectory("scaled"+demo_name, t_scale=1)
    print(max_abs_jerk(scaled_dmp))

    smooth_dmp = TrajectoryStock()
    smooth_dmp.import_from_dmp_joint_trajectory("smooth"+demo_name, t_scale=1)
    print(max_abs_jerk(smooth_dmp))

    # smooth_dmp = TrajectoryStock()
    # smooth_dmp.import_from_dmp_joint_trajectory("smooth"+demo_name, t_scale=0)

    # original_traj = TrajectoryStock()
    # original_traj.import_from_lfd_storage("filter"+demo_name, t_scale=0)
    # print(max_abs_jerk(original_traj))
    # print(duration(original_traj))

    # smooth_traj = TrajectoryStock()
    # smooth_traj.import_from_pydrake("smooth"+demo_name, t_scale=0)
    # print(max_abs_jerk(smooth_traj))
    # print(duration(smooth_traj))

if __name__ == '__main__':

    rospy.init_node('icra24')
    os.chdir(rospy.get_param("~working_dir"))

    demo_name = "picknplace0"

    # original_traj = TrajectoryStock()
    # original_traj.import_from_lfd_storage("filter"+demo_name, t_scale=0)   
    # original_traj.plot() 

    # demo_name = "picknplaceee0"
    # tol_an = tolerance_analysis2()

    # velocity_adjustment_analysis()

    tolerance_analysis()

    # dmp_analysis()


    # traj_analysis()


    # rospy.spin()