#!/usr/bin/env python3

import os
import rospy

from lfd_smoother.plotting.trajectory import TrajectoryStock

import actionlib
from control_msgs.msg import FollowJointTrajectoryAction, FollowJointTrajectoryGoal

def send_trajectory(joint_trajectory):
    # Initialize the action client
    client = actionlib.SimpleActionClient('/position_joint_trajectory_controller/follow_joint_trajectory', 
                                          FollowJointTrajectoryAction)
    
    # Wait for the action server to come up
    rospy.loginfo("Waiting for action server...")
    client.wait_for_server()
    rospy.loginfo("Action server detected!")

    # Create and populate the goal message
    goal = FollowJointTrajectoryGoal()
    goal.trajectory = joint_trajectory

    # Send the goal
    client.send_goal(goal)
    rospy.loginfo("Trajectory sent!")

    # Optionally, wait for the action to complete
    client.wait_for_result()
    return client.get_result()


if __name__ == '__main__':

    rospy.init_node('execute_trajectory')
    os.chdir(rospy.get_param("~working_dir"))
    demo_name = rospy.get_param("~demo_name")
    traj_arg = rospy.get_param("~traj_arg")
    traj_duration = int(rospy.get_param("~traj_duration"))

    if traj_arg == "refined":
        correct_traj = TrajectoryStock()
        correct_traj.import_from_pydrake("correct"+demo_name, t_scale=traj_duration)
        traj = correct_traj.to_joint_trajectory()
    elif traj_arg == "smooth":
        smooth_traj = TrajectoryStock()
        smooth_traj.import_from_pydrake("smooth"+demo_name, t_scale=traj_duration)
        traj = smooth_traj.to_joint_trajectory()
    elif traj_arg == "original":
        original_traj = TrajectoryStock()
        original_traj.import_from_lfd_storage("filter"+demo_name, t_scale=traj_duration)
        traj = original_traj.to_joint_trajectory(withvelacc=False)

    send_trajectory(traj)
