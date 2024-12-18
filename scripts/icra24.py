#!/usr/bin/env python3

import os
import rospy
import numpy as np

# from pydrake.all import *

from lfd_smoother.plotting.trajectory import TrajectoryStock
from lfd_smoother.plotting.dmp import DMPAnalysis
from lfd_smoother.plotting.jerk import JerkAnalysis
from lfd_smoother.plotting.refinement_phase import ToleranceAnalysis
from lfd_smoother.plotting.frequency import FrequencyAnalysis
from lfd_smoother.plotting.torque import TorqueAnalysis


# f = FrequencyAnalysis()
# f.run(demo_name)

# torque_analysis = TorqueAnalysis()
# print(torque_analysis.run(original_traj))

# cartesian_analyser = CartesianAnalysis()
# cartesian_analyser.from_trajectory(smooth_traj)
# cartesian_analyser.plot()



class Plot:

    def __init__(self, demo_name, robot_type="fr3") -> None:
        self.demo_name  = demo_name
        self.robot_type = robot_type
        self._add_original_traj()
        self._add_smooth_traj()
        self._add_refined_traj()
        self._add_dmps_traj()

    def _add_original_traj(self):
        try:
            self.original_traj = TrajectoryStock()
            self.original_traj.import_from_lfd_storage("filter"+self.demo_name, t_scale=0)
            self.original_traj.add_cartesian_analysis(robot_type=self.robot_type)

            self.original_traj_n = TrajectoryStock()
            self.original_traj_n.import_from_lfd_storage("filter"+self.demo_name, t_scale=1)
            self.original_traj_n.add_cartesian_analysis(robot_type=self.robot_type)
        except:
            print("failed to load original trajectory")

    def _add_smooth_traj(self):
        try:
            self.smooth_traj = TrajectoryStock()
            self.smooth_traj.import_from_pydrake("smooth"+self.demo_name, t_scale=0)
            self.smooth_traj.add_cartesian_analysis(robot_type=self.robot_type)

            self.smooth_traj_n = TrajectoryStock()
            self.smooth_traj_n.import_from_pydrake("smooth"+self.demo_name, t_scale=1)
            self.smooth_traj_n.add_cartesian_analysis(robot_type=self.robot_type)
        except:
            print("failed to load smooth trajectory")

    def _add_refined_traj(self):
        try:
            self.refined_traj = TrajectoryStock()
            self.refined_traj.import_from_pydrake("correct"+self.demo_name, t_scale=0)
            self.refined_traj.add_cartesian_analysis(robot_type=self.robot_type)        

            self.refined_traj_n = TrajectoryStock()
            self.refined_traj_n.import_from_pydrake("correct"+self.demo_name, t_scale=1)
            self.refined_traj_n.add_cartesian_analysis(robot_type=self.robot_type) 
        except:
            print("failed to load refined trajectory")

    def _add_dmps_traj(self):
        try:
            self.original_dmp = TrajectoryStock()
            self.original_dmp.import_from_dmp_joint_trajectory("filter"+self.demo_name, t_scale=0)
            self.original_dmp.add_cartesian_analysis(robot_type=self.robot_type)

            # self.scaled_dmp = TrajectoryStock()
            # self.scaled_dmp.import_from_dmp_joint_trajectory("scaled"+self.demo_name, t_scale=0)
            # self.scaled_dmp.add_cartesian_analysis(robot_type=self.robot_type)

            self.smooth_dmp = TrajectoryStock()
            self.smooth_dmp.import_from_dmp_joint_trajectory("smooth"+self.demo_name, t_scale=0)
            self.smooth_dmp.add_cartesian_analysis(robot_type=self.robot_type) 

            self.original_dmp_n = TrajectoryStock()
            self.original_dmp_n.import_from_dmp_joint_trajectory("filter"+self.demo_name, t_scale=1)
            self.original_dmp_n.add_cartesian_analysis(robot_type=self.robot_type)

            # self.scaled_dmp_n = TrajectoryStock()
            # self.scaled_dmp_n.import_from_dmp_joint_trajectory("scaled"+self.demo_name, t_scale=1)
            # self.scaled_dmp_n.add_cartesian_analysis(robot_type=self.robot_type)

            self.smooth_dmp_n = TrajectoryStock()
            self.smooth_dmp_n.import_from_dmp_joint_trajectory("smooth"+self.demo_name, t_scale=1)
            self.smooth_dmp_n.add_cartesian_analysis(robot_type=self.robot_type) 

        except:
            print("failed to load dmp trajectory")

    def refinement_phase(self):
        tol_an = ToleranceAnalysis()
        tol_an.import_data("smooth" + demo_name)
        # tol_an.plot_traj_with_tols(self.refined_traj)
        # tol_an.plot_x_with_tols(self.refined_traj)
        # tol_an.plot_t_s_command()
        tol_an.plot_waypoints(self.smooth_traj.ts[-1], self.refined_traj.ts[-1])
        # tol_an.plot_tols_only()

    def jerk_analysis(self):
        jerk_analysis = JerkAnalysis()
        jerk_analysis.plot_with_high_jerk(self.original_traj.ts, self.original_traj.ys[:,1], self.original_traj_n.yddds[:,1])
        jerk_analysis.plot_with_low_jerk(self.smooth_traj.ts, self.smooth_traj.ys[:,1], self.smooth_traj_n.yddds[:,1])

    def tolerance_comparison(self):
        tol_an = ToleranceAnalysis()
        tol_an.import_data("smooth" + demo_name)
        tol_an.plot_low_high_tol(self.smooth_traj, self.refined_traj)

    def dmp_analysis(self):
        dmp_an = DMPAnalysis()
        dmp_an.plot_3d(self.original_dmp, self.smooth_dmp)
        # dmp_an.plot_vel_acc_kin_lims(self.scaled_dmp, self.smooth_dmp)
        # dmp_an.plot_compare_jerks(self.scaled_dmp,self.smooth_dmp)


        # dmp_an.plot_abs_jerk(self.original_dmp, self.smooth_dmp, self.scaled_dmp)
        # dmp_an.plot_with_kin_lims(self.scaled_dmp)
        # dmp_an.plot_with_kin_lims(self.smooth_dmp)
    
    def max_abs_jerk(self, traj : TrajectoryStock):
        return np.max(np.abs(traj.yddds))

    def duration(self, traj:TrajectoryStock):
        return traj.ts[-1]
    

def print_demo_info(demo_names):
    for demo_name in demo_names:
        plot = Plot(demo_name, robot_type="yumi_l")
        
        print(f"Demo: {demo_name}")
        print("original_traj duration", plot.duration(plot.original_traj))
        print("smooth_traj duration", plot.duration(plot.smooth_traj))
        print("original_traj MANJ", plot.max_abs_jerk(plot.original_traj_n))
        print("smooth_traj MANJ", plot.max_abs_jerk(plot.smooth_traj_n))
        print("original_dmp MANJ", plot.max_abs_jerk(plot.original_dmp_n))
        print("smooth_dmp MANJ", plot.max_abs_jerk(plot.smooth_dmp_n))
        print(plot.max_abs_jerk(plot.original_traj_n), plot.max_abs_jerk(plot.smooth_traj_n),
              plot.max_abs_jerk(plot.original_dmp_n), plot.max_abs_jerk(plot.smooth_dmp_n))
        print()
        # print in this format: plot.duration(plot.original_traj) & \textbf{plot.duration(plot.smooth_traj)} & & plot.duration(plot.original_traj) & \textbf{plot.duration(plot.smooth_traj)}
        print(f"{plot.duration(plot.original_traj):.2f} & {plot.duration(plot.smooth_traj):.2f} &  & {plot.duration(plot.original_traj):.2f} & {plot.duration(plot.smooth_dmp):.2f}")
        print(f"{plot.max_abs_jerk(plot.original_traj_n):.2f} & {plot.max_abs_jerk(plot.smooth_traj_n):.2f} &  & {plot.max_abs_jerk(plot.original_dmp_n):.2f} & {plot.max_abs_jerk(plot.smooth_dmp_n):.2f}")
        print()




if __name__ == '__main__':

    rospy.init_node('icra24')
    os.chdir(rospy.get_param("~working_dir"))
    demo_name = rospy.get_param("~demo_name")
    plot_arg = rospy.get_param("~plot_arg")

    plot = Plot(demo_name)

    # demo_names_reaching = ["picknplace0", "picknplaceee0", "frreachone0", "frreachtwo0",
    #                 "frreachfour0", "frreachfive0",
    #                  "frreachsix0", "frreachseven0", "frreacheight0",
    #                  "frreachnine0", "frreachten0"]

    # demo_names_moving = ["frplace0","frmoveone0", "frmovetwo0", "frmovethree0",
    #                       "frmovefour0", "frmovefive0", "frmovesix0",
    #                         "frmoveeight0", "frmovenine0", "frmoveten0"]
    
    # demo_names = ["picknplace0","picknplaceee0", "frreachfive0", "frreachsix0",
    #                       "frreacheight0", "frplace0", "frmoveone0",
    #                         "frmovefive0", "frmovesix0", "frmoveten0"]

    # demo_names_yumi = ["ylhometoscrew0","ylscrewmount0", "ylscrewtoring0", "ylringmount0"]
    
    # print_demo_info(demo_names_yumi)

    # plot.original_traj.cartesian.plot()
    
    
    plot.original_traj.plot()

    # plot.refinement_phase()
    # plot.dmp_analysis()

    # if plot_arg == "dmp":
    #     plot.dmp_analysis()
    # elif plot_arg == "refinement":
    #     plot.refinement_phase()
    # elif plot_arg == "jerk":
    #     plot.jerk_analysis()
    # elif plot_arg == "tolerance":
    #     plot.tolerance_comparison()





    # refined_traj = TrajectoryStock()
    # refined_traj.import_from_pydrake("correct"+demo_name, t_scale=0)
    # traj = refined_traj.to_joint_trajectory()


    # smooth_traj = TrajectoryStock()
    # smooth_traj.import_from_pydrake("smooth"+demo_name, t_scale=10)
    # traj = smooth_traj.to_joint_trajectory()

    # original_traj = TrajectoryStock()
    # original_traj.import_from_lfd_storage("filter"+demo_name, t_scale=0)
    # traj = original_traj.to_joint_trajectory(withvelacc=False)

    # send_trajectory(traj)
